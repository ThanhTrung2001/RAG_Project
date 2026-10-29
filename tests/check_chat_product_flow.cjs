// Exercise the actual ask() wiring with API/DOM fixtures; no browser or Ollama.
// Run from repo root: node tests/check_chat_product_flow.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const nodes = new Map();
const node = () => ({
  value: '', checked: false, disabled: false, textContent: '',
  classList: { add() {}, remove() {} }, appendChild() {}, prepend() {},
});
const context = vm.createContext({
  URLSearchParams, console,
  URL: {createObjectURL: () => 'blob:fixture'},
  FormData: class {
    constructor() { this.fields = new Map(); }
    append(key, value) { this.fields.set(key, value); }
    get(key) { return this.fields.get(key); }
  },
  fileFixture: {name: 'product.png'},
  document: { getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, node());
    return nodes.get(id);
  }, createElement: node },
});
const html = fs.readFileSync(path.join(__dirname, '../frontend/chat.html'), 'utf8');
const source = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)][0][1];
vm.runInContext(source.replace(/\nloadRerankers\(\);\s*$/, '\n'), context);
vm.runInContext(`
  var renderedList, renderedCitations, openedDetail;
  appendBubble = () => ({classList: {remove() {}, add() {}}, appendChild() {}, prepend() {}});
  selectedReranker = () => 'none';
  renderSources = data => { renderedList = data.sources; };
  renderCitedProducts = (_, cited) => { renderedCitations = cited; };
  showProductDetail = async (product, number) => { openedDetail = {id: product.id, number}; };
`, context);
const product = id => ({id, caption: `Product ${id}`, image_path: `${id}.jpg`});
const [a, b, c, d] = [10, 20, 30, 40].map(product);
const initial = [a, b];
const replacement = [c, d];
const steps = [
  {query: 'Find products', intent: 'search', sources: initial, options: initial, sent: null},
  {query: 'Product 1 details', intent: 'detail', sources: [a], options: initial, sent: null, number: 1},
  {query: 'Product 2 details', intent: 'detail', sources: [b], options: initial, sent: '10', number: 2},
  {query: 'Weight?', intent: 'detail', sources: [b], options: initial, sent: '20', number: 2},
  {query: 'Compare product 2 and product 1', intent: 'detail', sources: [b, a], options: initial, sent: '20'},
  {query: 'Find coffee makers', intent: 'search', sources: replacement, options: replacement, sent: null},
  {query: 'Product 1 details', intent: 'detail', sources: [c], options: replacement, sent: null, number: 1},
  {query: 'Find one product', intent: 'search', sources: [b], options: [b], sent: '30', number: 1},
  {query: 'Capacity?', intent: 'detail', sources: [b], options: [b], sent: '20', number: 1},
  {query: 'Recommend one of these', intent: 'search', sources: initial, options: initial,
   sent: '20', focus: 20, number: 2, answer: 'I recommend only [2].', cited: 20},
  {query: 'Capacity?', intent: 'detail', sources: [b], options: initial, sent: '20', number: 2},
  {query: 'Find something else', intent: 'search', sources: [], options: [], sent: '20'},
  {query: 'air fryer', image: true, intent: 'search', sources: initial, options: initial,
   focus: 20, number: 2, answer: 'I recommend [2].', cited: 20},
  {query: 'Capacity?', intent: 'detail', sources: [b], options: initial, sent: '20', number: 2},
  {query: '', image: true, intent: 'search', sources: [c], options: [c], number: 1},
  {query: 'Weight?', intent: 'detail', sources: [c], options: [c], sent: '30', number: 1},
];

(async () => {
  for (const mode of [false, true]) {
    vm.runInContext('selectedProductId = null;', context);
    for (const step of steps) {
      context.document.getElementById('question').value = step.query;
      context.document.getElementById('agentic').checked = mode;
      vm.runInContext(`attachedImage = ${step.image ? 'fileFixture' : 'null'};`, context);
      let called = false;
      context.fetch = async (url, request) => {
        called = true;
        if (step.image) {
          assert.equal(url, '/api/v1/ask_image');
          assert.equal(request.method, 'POST');
          assert.equal(request.body.get('file'), context.fileFixture);
          assert.equal(request.body.get('q'), step.query);
          assert.equal(request.body.get('agentic'), mode);
          assert.equal(request.body.get('k'), 5);
          assert.ok(request.body.get('session_id'));
        } else {
          const params = new URL(url, 'http://local.test').searchParams;
          assert.equal(params.get('selected_product_id'), step.sent);
          assert.equal(params.get('agentic'), String(mode));
        }
        return {ok: true, json: async () => ({
          answer: step.answer ?? (step.sources.length ? 'Catalog answer [1].' : 'No products.'),
          intent: step.intent, sources: step.sources,
          ...(step.focus !== undefined ? {focused_product_id: step.focus} : {}),
          ...(step.intent === 'detail' ? {product_options: step.options} : {}),
        })};
      };
      await vm.runInContext('ask()', context);
      assert.ok(called);
      assert.equal(vm.runInContext('attachedImage', context), null);
      assert.deepEqual(Array.from(context.renderedList, p => p.id), step.options.map(p => p.id));
      // Local [1] refers to the answered product, even if it is search option 2.
      if (step.sources.length) {
        assert.equal(context.renderedCitations[0].source.id, step.cited ?? step.sources[0].id);
      } else {
        assert.equal(context.renderedCitations.length, 0);
      }
      if (step.number) {
        assert.equal(context.openedDetail.id, step.focus ?? step.sources[0].id);
        assert.equal(context.openedDetail.number, step.number);
      }
      assert.equal(context.document.getElementById('btnSend').disabled, false);
    }
  }
  console.log(`PASS: ${steps.length * 2} UI wiring turns (Naive + Agentic): text/image/image+text, stable list, verbal switch, auto-focus, local citations, compare, new search.`);
})().catch(error => { console.error(error); process.exitCode = 1; });
