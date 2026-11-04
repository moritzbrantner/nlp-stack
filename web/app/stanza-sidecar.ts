/**
 * Opt-in local Stanza sidecar (`sidecars/stanza/serve`). The browser builds the
 * request in Rust/Wasm (`analysis.stanza-request`), posts it to the sidecar and hands
 * the unchanged answer back to Rust/Wasm (`linguistics.mode: "stanza"`), which checks
 * versions, sentence identity and spans. There is no fallback: an unreachable sidecar
 * or a missing model stops the analysis with its reason.
 */

export const defaultStanzaSidecarUrl = "http://127.0.0.1:8765";
const sidecarTimeoutMs = 180_000;

type JsonRecord = Record<string, unknown>;

export class StanzaSidecarError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StanzaSidecarError";
  }
}

/** `http://host:port` without a trailing slash, or an error for anything else. */
export function normalizeStanzaSidecarUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, "");
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new StanzaSidecarError(`"${value}" is not a URL. Use the sidecar address, for example ${defaultStanzaSidecarUrl}.`);
  }
  if (url.protocol !== "http:" || url.pathname !== "/" || url.search || url.hash) {
    throw new StanzaSidecarError(`Use the sidecar's http://host:port address (for example ${defaultStanzaSidecarUrl}), not "${value}".`);
  }
  return `${url.protocol}//${url.host}`;
}

/** Posts `request` (the `analysis.stanza-request` value) to the sidecar's `/v1/analyze`. */
export async function analyzeWithStanzaSidecar(
  baseUrl: string,
  request: unknown,
  { signal, fetchImpl = fetch }: { signal?: AbortSignal; fetchImpl?: typeof fetch } = {},
): Promise<JsonRecord> {
  const url = `${normalizeStanzaSidecarUrl(baseUrl)}/v1/analyze`;
  const timeout = AbortSignal.timeout(sidecarTimeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;
  let response: Response;
  try {
    response = await fetchImpl(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: combined,
    });
  } catch (caught) {
    if (signal?.aborted) throw caught;
    if (timeout.aborted) {
      throw new StanzaSidecarError(`The Stanza sidecar at ${baseUrl} did not answer within ${sidecarTimeoutMs / 1000} seconds.`);
    }
    throw new StanzaSidecarError(
      `The Stanza sidecar at ${baseUrl} is not reachable. Start it with \`sidecars/stanza/serve\` (after \`bash sidecars/stanza/setup.sh\`). No fallback syntax is shown.`,
    );
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new StanzaSidecarError(`The Stanza sidecar at ${baseUrl} answered HTTP ${response.status} without JSON.`);
  }
  const record = body && typeof body === "object" && !Array.isArray(body) ? (body as JsonRecord) : null;
  if (!response.ok) {
    const error = record?.error && typeof record.error === "object" ? (record.error as JsonRecord) : null;
    const kind = typeof error?.kind === "string" ? error.kind : `HTTP ${response.status}`;
    const message = typeof error?.message === "string" ? error.message : "no error message";
    throw new StanzaSidecarError(`Stanza sidecar: ${kind}: ${message}`);
  }
  if (!record) throw new StanzaSidecarError("The Stanza sidecar answered with something other than a JSON object.");
  return record;
}
