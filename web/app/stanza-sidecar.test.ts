import { describe, expect, it, vi } from "vitest";

import { analyzeWithStanzaSidecar, normalizeStanzaSidecarUrl, StanzaSidecarError } from "./stanza-sidecar";

const request = { sentences: [{ id: 0, language: "de", text: "Er geht zum Bahnhof." }] };

function jsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("Stanza sidecar client", () => {
  it("accepts only an http://host:port address", () => {
    expect(normalizeStanzaSidecarUrl(" http://127.0.0.1:8765/ ")).toBe("http://127.0.0.1:8765");
    expect(() => normalizeStanzaSidecarUrl("127.0.0.1:8765")).toThrow(StanzaSidecarError);
    expect(() => normalizeStanzaSidecarUrl("http://127.0.0.1:8765/v1/analyze")).toThrow(/host:port/);
  });

  it("posts the Rust-built request unchanged and returns the answer", async () => {
    const answer = { protocol: 1, stanza_version: "1.15.0", sentences: [] };
    const fetchImpl = vi.fn(async () => jsonResponse(200, answer));
    await expect(analyzeWithStanzaSidecar("http://127.0.0.1:8765", request, { fetchImpl })).resolves.toEqual(answer);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8765/v1/analyze",
      expect.objectContaining({ method: "POST", body: JSON.stringify(request) }),
    );
  });

  it("reports an unreachable sidecar without falling back", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    await expect(analyzeWithStanzaSidecar("http://127.0.0.1:8765", request, { fetchImpl })).rejects.toThrow(
      /not reachable\. Start it with `sidecars\/stanza\/serve`/,
    );
  });

  it("surfaces the sidecar's typed error", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(503, { error: { kind: "model_missing", message: "pinned Stanza models for `de` are not usable", language: "de" } }),
    );
    await expect(analyzeWithStanzaSidecar("http://127.0.0.1:8765", request, { fetchImpl })).rejects.toThrow(
      "Stanza sidecar: model_missing: pinned Stanza models for `de` are not usable",
    );
  });
});
