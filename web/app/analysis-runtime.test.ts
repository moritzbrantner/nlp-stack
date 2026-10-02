import { afterEach, expect, it, vi } from "vitest";

afterEach(() => {
  document.getElementById("nlp-stack-text-analysis-runtime")?.remove();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  vi.resetModules();
});

it("cancels during script loading without creating a late worker session", async () => {
  const { loadAnalysisRuntime } = await import("./analysis-runtime");
  const controller = new AbortController();
  const pending = loadAnalysisRuntime(controller.signal);
  const failure = expect(pending).rejects.toMatchObject({ name: "AbortError" });
  controller.abort();
  await failure;
  const createSession = vi.fn();
  vi.stubGlobal("nlpStackTextAnalysis", { ready: Promise.resolve({ createSession }) });
  window.dispatchEvent(new Event("nlp-stack-text-analysis-ready"));
  // A second caller waits for the same registration; only it receives a session.
  await loadAnalysisRuntime(new AbortController().signal);
  expect(createSession).toHaveBeenCalledTimes(1);
});

it("reports a script error and permits a new loading attempt", async () => {
  const { loadAnalysisRuntime } = await import("./analysis-runtime");
  const first = expect(loadAnalysisRuntime(new AbortController().signal)).rejects.toThrow("Failed to load");
  document.getElementById("nlp-stack-text-analysis-runtime")?.dispatchEvent(new Event("error"));
  await first;
  const second = expect(loadAnalysisRuntime(new AbortController().signal)).rejects.toThrow("Failed to load");
  document.getElementById("nlp-stack-text-analysis-runtime")?.dispatchEvent(new Event("error"));
  await second;
});

it("bounds a runtime script that never registers", async () => {
  vi.useFakeTimers();
  const { loadAnalysisRuntime } = await import("./analysis-runtime");
  const failure = expect(loadAnalysisRuntime(new AbortController().signal)).rejects.toThrow("30 seconds");
  await vi.advanceTimersByTimeAsync(30_000);
  await failure;
  expect(document.getElementById("nlp-stack-text-analysis-runtime")).toBeNull();
});
