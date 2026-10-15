const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const crypto = require('node:crypto');

function workspace({ refresh = false } = {}) {
  let tokenSeen;
  const seen = new Promise((resolve) => { tokenSeen = resolve; });
  const text = { style: {}, value: '', set textContent(value) { this.value = value; tokenSeen(value); } };
  const final = { innerHTML: '' };
  const message = { querySelector: () => ({ nextElementSibling: text }), lastElementChild: final };
  const input = { value: '退款多久到账？' };
  const button = { dataset: {} };
  const messages = { innerHTML: '', scrollHeight: 10, querySelectorAll: () => [] };
  const requests = [];
  let controller;
  const body = new ReadableStream({ start(value) { controller = value; } });
  const context = vm.createContext({
    localStorage: { getItem: () => '' },
    document: {
      querySelector: (selector) => ({ '#chat-messages': messages, '#agent-prompt': input,
        '#agent-form button[type=submit]': button })[selector] || null,
      querySelectorAll: () => [],
    },
    crypto, AbortController, TextDecoder,
    fetch: async (url, options) => {
      requests.push({ url, options });
      if (refresh && requests.length === 1) return {
        ok: false, status: 401, headers: { get: () => 'application/json' },
        json: async () => ({ error: { message: 'expired' } }),
      };
      options.signal.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
      return { ok: true, status: 200, body };
    },
  });
  const source = fs.readFileSync(path.join(__dirname, '../backend/app/static/app.js'), 'utf8');
  vm.runInContext(source.replace(/\nboot\(\);\s*$/, ''), context);
  Object.assign(context, {
    addMessage: () => message, setLoading: () => {}, renderRunDetails: () => {},
    showToast: () => { throw new Error('unexpected toast'); },
    loadApprovals: async () => {}, loadTickets: async () => {},
    refreshAccessToken: async () => 'renewed',
  });
  vm.runInContext("state.userId='user'; state.orgId='org'; state.refreshToken='refresh'", context);
  return {
    context, seen, text, final, requests,
    run: () => context.runAgent({ preventDefault() {}, currentTarget: { querySelector: () => button } }),
    emit: (event) => {
      const bytes = new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
      // Exercise frames and Chinese UTF-8 characters split at every byte boundary.
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
    },
    finish: () => controller.close(),
  };
}

test('tokens render before completion; refresh preserves the idempotency key and SSE mode', async () => {
  const ui = workspace({ refresh: true });
  const task = ui.run();
  ui.emit({ type: 'answer.delta', text: '退款<img src=x onerror=alert(1)>' });
  assert.equal(await ui.seen, '退款<img src=x onerror=alert(1)>');
  assert.equal(ui.final.innerHTML, '');
  assert.equal(ui.requests.length, 2);
  assert.equal(ui.requests[0].options.body, ui.requests[1].options.body);
  assert.ok(JSON.parse(ui.requests[0].options.body).idempotency_key);
  ui.emit({ type: 'run.result', run: {
    id: 'run', status: 'succeeded', trace_id: 'trace', answer: '退款<img src=x onerror=alert(1)>',
  } });
  ui.finish();
  await task;
  assert.ok(ui.final.innerHTML.includes('&lt;img'));
  assert.ok(!ui.final.innerHTML.includes('<img'));
  assert.equal(vm.runInContext('state.agentAbort', ui.context), null);
});

test('switching workspace cancels the stream and prevents stale UI updates', async () => {
  const ui = workspace();
  const task = ui.run();
  ui.emit({ type: 'answer.delta', text: 'partial' });
  assert.equal(await ui.seen, 'partial');
  vm.runInContext('resetAgentChat()', ui.context);
  await task;
  assert.equal(ui.requests[0].options.signal.aborted, true);
  assert.equal(ui.final.innerHTML, '');
});
