'use strict';
// One transport contract for the local API and the static original playing table.
class HnefTransport {
  constructor({baseURL, mode = 'http', workerFactory, fetcher, startupTimeout = 60000} = {}) {
    this.baseURL = baseURL;
    this.mode = mode;
    this.workerFactory = workerFactory || (url => new Worker(url, {type: 'module'}));
    this.fetcher = fetcher || ((...args) => fetch(...args));
    this.startupTimeout = startupTimeout;
    this.generation = 0;
    this.sequence = 0;
    this.pending = new Map();
    this.queue = Promise.resolve();
    this.status = '';
    this.lastReplay = null;
  }
  url(path = '') { return new URL(path.replace(/^\//, ''), this.baseURL).href; }
  setStatus(text) { this.status = text; this.onStatus?.(text); }
  stop(error = new Error('Calculation cancelled. Your saved game is unchanged.')) {
    this.generation += 1;
    this.worker?.terminate();
    this.worker = null;
    this.ready = false;
    this.starting = null;
    for (const {reject, timer} of this.pending.values()) {
      clearTimeout(timer);
      reject(error);
    }
    this.pending.clear();
    this.controller?.abort(error);
    this.setStatus('');
  }
  cancel() { this.stop(); }
  call(kind, payload, timeout) {
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      const timer = setTimeout(() => this.stop(new Error(
        'Calculation timed out. Your saved game is unchanged. Retry to restart the engine.'
      )), timeout);
      this.pending.set(id, {resolve, reject, timer});
      try { this.worker.postMessage({id, kind, payload}); }
      catch (error) { this.stop(error); }
    });
  }
  async ensureReady() {
    if (this.ready) return;
    if (this.starting) return this.starting;
    const generation = this.generation;
    const worker = this.workerFactory(this.url('table-worker.mjs'));
    this.worker = worker;
    worker.onmessage = ({data}) => {
      if (generation !== this.generation || worker !== this.worker) return;
      const pending = this.pending.get(data.id);
      if (!pending) return;
      if (data.fatal) { this.stop(new Error(data.fatal)); return; }
      clearTimeout(pending.timer);
      this.pending.delete(data.id);
      pending.resolve(data);
    };
    worker.onerror = event => {
      if (generation !== this.generation) return;
      event.preventDefault?.();
      this.stop(new Error('The game engine stopped. Your saved game is unchanged. Retry to restart it.'));
    };
    worker.onmessageerror = () => {
      if (generation === this.generation) this.stop(new Error('The engine response could not be read. Retry to restart it.'));
    };
    this.setStatus('Loading game engine…');
    const starting = (async () => {
      await this.call('init', {}, this.startupTimeout);
      if (this.lastReplay) {
        this.setStatus('Restoring saved position…');
        const recovered = await this.call('game', {operation: 'import', replay: this.lastReplay}, this.startupTimeout);
        if (!recovered.ok) {
          this.stop(new Error('The saved position could not be restored. Export your replay before starting a new game.'));
          throw new Error('Saved position recovery failed.');
        }
      }
      if (generation !== this.generation) throw new Error('Calculation cancelled.');
      this.ready = true;
      this.setStatus('');
    })();
    this.starting = starting;
    try { await starting; }
    finally { if (this.starting === starting) this.starting = null; }
  }
  request(kind, payload, {timeout = 30000, remember = true} = {}) {
    // Snapshot at submission: later form edits cannot alter queued requests.
    const body = JSON.stringify(payload);
    const generation = this.generation;
    const operation = this.queue.then(async () => {
      if (generation !== this.generation) throw new Error('Calculation cancelled. Please retry.');
      if (!['game', 'analyze'].includes(kind)) throw new Error('Unknown request.');
      if (new TextEncoder().encode(body).length > 131072) {
        return {ok: false, status: 400, json: async () => ({error: 'Invalid request size'})};
      }
      let response;
      if (this.mode === 'worker') {
        await this.ensureReady();
        this.setStatus(kind === 'analyze' ? 'Analyzing…' : 'Calculating…');
        const message = await this.call(kind, JSON.parse(body), timeout);
        response = {ok: message.ok, status: message.status, json: async () => message.data};
      } else {
        const controller = new AbortController();
        this.controller = controller;
        const timer = setTimeout(() => controller.abort(new Error('The local server timed out. Please retry.')), timeout);
        try {
          // Read the body under the deadline too; a response header alone is not completion.
          const result = await this.fetcher(this.url(`api/${kind}`), {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body, signal: controller.signal,
          });
          const data = await result.json();
          response = {ok: result.ok, status: result.status, json: async () => data};
        } finally { clearTimeout(timer); this.controller = null; }
      }
      if (generation !== this.generation) throw new Error('Calculation cancelled. Please retry.');
      if (this.mode === 'worker' && kind === 'game' && remember && response.ok) {
        this.lastReplay = structuredClone((await response.json()).replay);
      }
      this.setStatus('');
      return response;
    });
    this.queue = operation.catch(() => {});
    return operation;
  }
}
if (typeof module !== 'undefined') module.exports = {HnefTransport};
if (typeof document !== 'undefined') {
  const script = document.currentScript;
  window.hnefTransport = new HnefTransport({
    baseURL: new URL('./', script.src).href, mode: script.dataset.mode || 'http',
  });
  addEventListener('pagehide', () => window.hnefTransport.cancel());
}
