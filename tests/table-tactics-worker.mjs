// Test-only runner, copied into the QA site, never included in build_table output.
self.onmessage = async () => {
  try {
    const {loadPyodide} = await import('../runtime/pyodide.mjs');
    const py = await loadPyodide({indexURL: new URL('../runtime/', self.location.href).href});
    const bundle = await (await fetch('../engine.json')).json();
    for (const [name, source] of Object.entries(bundle.files)) {
      const path = '/engine/' + name;
      py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
      py.FS.writeFile(path, source);
    }
    py.FS.writeFile('/engine/table_parity.py', await (await fetch('./table_parity.py')).text());
    py.FS.writeFile('/engine/forts.json', await (await fetch('./exit-forts.json')).text());
    const result = py.runPython(`import sys, json
sys.path.insert(0, '/engine')
from table_parity import tactical_results
json.dumps(tactical_results(json.load(open('/engine/forts.json'))))`);
    self.postMessage({result: JSON.parse(result)});
  } catch (error) { self.postMessage({error: String(error.stack || error)}); }
};
