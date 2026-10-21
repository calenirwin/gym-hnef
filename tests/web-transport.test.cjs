const {test} = require('node:test');
const assert = require('node:assert/strict');
const {HnefTransport} = require('../src/hnef/web/transport.js');
const tick = () => new Promise(resolve => setImmediate(resolve));
function harness() {
  const workers = [];
  const transport = new HnefTransport({baseURL:'https://example.test/gym-hnef/',mode:'worker',startupTimeout:2000,
    workerFactory: url => {
      assert.equal(url,'https://example.test/gym-hnef/table-worker.mjs');
      const worker = {messages:[], terminated:false,
        postMessage(message) { this.messages.push(message); },
        terminate() { this.terminated=true; },
        reply(message, data) { this.onmessage({data:{id:message.id,...data}}); },
      };
      workers.push(worker); return worker;
    }});
  return {transport,workers};
}
async function ready(worker) { worker.reply(worker.messages[0],{ok:true}); await tick(); }
const success = replay => ({ok:true,status:200,data:{replay}});

test('serializes requests, snapshots input and survives validation errors', async () => {
  const {transport:t,workers}=harness();
  const payload={actions:[1]};
  const a=t.request('game',payload), b=t.request('analyze',{iterations:8});
  payload.actions.push(2);
  await tick(); const w=workers[0]; await ready(w);
  assert.deepEqual(w.messages.map(m=>m.kind),['init','game']);
  assert.deepEqual(w.messages[1].payload,{actions:[1]});
  w.reply(w.messages[1],{ok:false,status:400,data:{error:'Illegal move'}});
  assert.equal((await a).status,400); await tick();
  assert.equal(w.messages[2].kind,'analyze');
  w.reply(w.messages[2],success({actions:[]})); await b; t.cancel();
});

test('cancellation rejects active and queued work; stale replies cannot update the next worker', async () => {
  const {transport:t,workers}=harness();
  const a=t.request('game',{}).catch(e=>e.message);
  const b=t.request('game',{}).catch(e=>e.message);
  await tick(); const old=workers[0]; await ready(old);
  const late=old.messages[1]; t.cancel();
  assert.match(await a,/cancelled/); assert.match(await b,/cancelled/);
  assert.equal(old.terminated,true);
  const next=t.request('game',{}); await tick(); const fresh=workers[1];
  old.reply(late,success({actions:['stale']}));
  assert.equal(t.lastReplay,null);
  await ready(fresh); fresh.reply(fresh.messages[1],success({actions:['new']}));
  await next; assert.deepEqual(t.lastReplay,{actions:['new']}); t.cancel();
});

test('timeout terminates worker, retains confirmed replay, validates it before retrying', async () => {
  const {transport:t,workers}=harness();
  const first=t.request('game',{}); await tick(); const w=workers[0]; await ready(w);
  w.reply(w.messages[1],success({actions:[1,2]})); await first;
  const timed=t.request('game',{operation:'ai'},{timeout:10});
  await assert.rejects(timed,/timed out/);
  assert.equal(w.terminated,true); assert.deepEqual(t.lastReplay,{actions:[1,2]});
  const retry=t.request('game',{operation:'ai'}); await tick(); const fresh=workers[1]; await ready(fresh);
  assert.deepEqual(fresh.messages[1].payload,{operation:'import',replay:{actions:[1,2]}});
  fresh.reply(fresh.messages[1],success({actions:[1,2]})); await tick();
  assert.equal(fresh.messages[2].payload.operation,'ai');
  fresh.reply(fresh.messages[2],success({actions:[1,2,3]})); await retry; t.cancel();
});

test('review requests never replace the last confirmed live replay', async () => {
  const {transport:t,workers}=harness();
  t.lastReplay={actions:[1,2]};
  const review=t.request('game',{actions:[]},{remember:false}); await tick(); const w=workers[0]; await ready(w);
  w.reply(w.messages[1],success({actions:[1,2]})); await tick();
  w.reply(w.messages[2],success({actions:[]})); await review;
  assert.deepEqual(t.lastReplay,{actions:[1,2]}); t.cancel();
});

test('startup failures reject and allow a fresh initialization without losing replay', async () => {
  const {transport:t,workers}=harness();
  const failed=t.request('game',{}); await tick(); workers[0].reply(workers[0].messages[0],{fatal:'Cannot load engine'});
  await assert.rejects(failed,/Cannot load engine/);
  const retry=t.request('game',{}); await tick(); const w=workers[1]; await ready(w);
  w.reply(w.messages[1],success({actions:[]})); await retry; t.cancel();
});

test('worker crash clears pending promises and request-size checks do not start a worker', async () => {
  const {transport:t,workers}=harness();
  assert.equal((await t.request('game',{text:'x'.repeat(131073)})).status,400);
  assert.equal(workers.length,0);
  const pending=t.request('game',{}); await tick(); workers[0].onerror({preventDefault(){}});
  await assert.rejects(pending,/engine stopped/); assert.equal(workers[0].terminated,true);
});

test('local HTTP mode preserves URL, body, validation response and body-reading deadline', async () => {
  const t = new HnefTransport({baseURL:'http://localhost:8000/',fetcher:async (url,options)=> {
    assert.equal(url,'http://localhost:8000/api/game');
    assert.deepEqual(JSON.parse(options.body),{actions:[]});
    return {ok:false,status:400,json:async()=>({error:'Bad replay'})};
  }});
  assert.deepEqual(await (await t.request('game',{actions:[]})).json(),{error:'Bad replay'});
  t.fetcher = async (_url, options) => ({ok:true,status:200,json:()=>new Promise((_,reject)=>{
    options.signal.addEventListener('abort',()=>reject(options.signal.reason));
  })});
  await assert.rejects(t.request('game',{}, {timeout:10}),/timed out/);
});
