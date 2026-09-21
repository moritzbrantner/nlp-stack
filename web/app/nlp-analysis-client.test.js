// @vitest-environment node
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createAnalysisWorkerRuntime } from "../public/nlp-analysis-client.js";

let workers;
beforeEach(() => {
  workers = [];
  vi.stubGlobal("Worker", class {
    listeners = new Map();
    requests = [];
    terminated = false;
    constructor() { workers.push(this); }
    addEventListener(type, listener) { this.listeners.set(type, listener); }
    postMessage(request) { this.requests.push(request); }
    terminate() { this.terminated = true; }
    reply(data) { this.listeners.get("message")({ data }); }
  });
});
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
const request = { operation: "analysis.document", input: { text: "Hello." } };

it("matches concurrent replies to their requests", async () => {
  const runtime = createAnalysisWorkerRuntime();
  const first = runtime.runOperation(request);
  const second = runtime.packageSurface();
  const [a, b] = workers[0].requests;
  workers[0].reply({ id: b.id, value: "surface" });
  workers[0].reply({ id: a.id, value: "document" });
  await expect(first).resolves.toBe("document");
  await expect(second).resolves.toBe("surface");
});

it("terminates active and queued work on cancellation and starts a fresh worker on retry", async () => {
  const runtime = createAnalysisWorkerRuntime();
  const controller = new AbortController();
  const tasks = [runtime.runOperation(request, { signal: controller.signal }), runtime.runOperation(request, { signal: controller.signal })];
  const results = Promise.allSettled(tasks);
  controller.abort();
  expect(workers[0].terminated).toBe(true);
  for (const result of await results) {
    expect(result.status).toBe("rejected");
    expect(result.reason.name).toBe("AbortError");
  }
  const retry = runtime.runOperation(request);
  workers[0].reply({ id: workers[1].requests[0].id, value: "stale" });
  workers[1].reply({ id: workers[1].requests[0].id, value: "fresh" });
  await expect(retry).resolves.toBe("fresh");
});

it("does not start already cancelled or oversized requests", async () => {
  const runtime = createAnalysisWorkerRuntime();
  const controller = new AbortController();
  controller.abort();
  await expect(runtime.runOperation(request, { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
  await expect(runtime.runOperation({ ...request, input: { text: "x".repeat(262145) } })).rejects.toThrow(/256 KiB/);
  expect(workers).toHaveLength(0);
});

it.each(["error", "messageerror"])("rejects pending work after worker %s and allows retry", async (kind) => {
  const runtime = createAnalysisWorkerRuntime();
  const task = runtime.runOperation(request);
  const failure = expect(task).rejects.toThrow(/worker/);
  workers[0].listeners.get(kind)({ message: "Analysis worker crashed" });
  await failure;
  expect(workers[0].terminated).toBe(true);
  const retry = runtime.runOperation(request);
  workers[1].reply({ id: workers[1].requests[0].id, value: "recovered" });
  await expect(retry).resolves.toBe("recovered");
});

it("bounds a worker that never replies", async () => {
  vi.useFakeTimers();
  const runtime = createAnalysisWorkerRuntime();
  const failure = expect(runtime.runOperation(request)).rejects.toThrow(/60 seconds/);
  await vi.advanceTimersByTimeAsync(60_000);
  await failure;
  expect(workers[0].terminated).toBe(true);
  expect(vi.getTimerCount()).toBe(0);
});

it("propagates operation errors without discarding successful replies", async () => {
  const runtime = createAnalysisWorkerRuntime();
  const failure = expect(runtime.runOperation(request)).rejects.toThrow("Invalid input");
  workers[0].reply({ id: workers[0].requests[0].id, error: "Invalid input" });
  await failure;
  const next = runtime.runOperation(request);
  workers[0].reply({ id: workers[0].requests[1].id, value: "ok" });
  await expect(next).resolves.toBe("ok");
});
