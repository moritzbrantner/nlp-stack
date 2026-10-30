export interface BrowserNerProducer {
  package: "@moritzbrantner/text-linguistics-wasm";
  module: "browser-ner";
  contractVersion: 1;
}

export interface BrowserNerModel {
  id: string;
  /** Pinned Hugging Face commit. */
  revision: string;
  baseModel: string;
  dtype: string;
  labels: string[];
  supportedLanguages: string[];
}

export interface BrowserNerCapabilities {
  producer: BrowserNerProducer;
  model: BrowserNerModel;
  runtime: { transformersModuleUrl: string };
}

export interface BrowserNerSegment {
  /** Defaults to the segment index. */
  id?: string | number;
  text: string;
  /** UTF-8 byte offset of `text` in the caller's source; defaults to 0. */
  byteStart?: number;
}

export interface BrowserNerEntity {
  segmentId: string | number;
  /** Model label without BIO prefix: PER, ORG, LOC or MISC. */
  label: string;
  /** Exact source slice: `segment.text.slice(start, end)`. */
  text: string;
  score: number;
  /** UTF-16 offsets within the segment text. */
  start: number;
  end: number;
  /** Absolute UTF-8 byte offsets (`segment.byteStart` + local). */
  byteStart: number;
  byteEnd: number;
}

export interface BrowserNerSegmentReport {
  id: string | number;
  entityCount: number;
  /** True when model input truncation left trailing text unlabeled. */
  truncated: boolean;
}

export interface BrowserNerResult extends BrowserNerCapabilities {
  status: "completed" | "unsupported-language";
  /** Primary language subtag that was requested, or null when unknown. */
  language: string | null;
  entities: BrowserNerEntity[];
  segments: BrowserNerSegmentReport[];
}

export interface BrowserNerProgress {
  stage: "model";
  /** Raw Transformers.js progress event (status, file, progress, loaded, total). */
  detail: Record<string, unknown>;
}

export interface BrowserNerRequestOptions {
  signal?: AbortSignal;
  onProgress?: (progress: BrowserNerProgress) => void;
}

export interface BrowserNerOptions {
  /** Overrides worker creation (for bundlers or tests). */
  createWorker?: () => Worker;
  /** Overrides the pinned Transformers.js module URL. */
  transformersModuleUrl?: string;
}

export interface BrowserNer {
  capabilities(): BrowserNerCapabilities;
  supportsLanguage(language: string | null | undefined): boolean;
  prepare(options?: BrowserNerRequestOptions): Promise<BrowserNerCapabilities>;
  analyze(
    request: { language: string | null | undefined; segments: BrowserNerSegment[] },
    options?: BrowserNerRequestOptions,
  ): Promise<BrowserNerResult>;
  dispose(): void;
}

export class BrowserNerError extends Error {
  stage: "model" | "analysis" | "worker";
}

export const BROWSER_NER_PRODUCER: BrowserNerProducer;
export const BROWSER_NER_MODEL: BrowserNerModel;
export const BROWSER_NER_TRANSFORMERS_MODULE_URL: string;
export function browserNerCapabilities(): BrowserNerCapabilities;
export function supportsBrowserNerLanguage(language: string | null | undefined): boolean;
export function entitiesFromTokenPredictions(
  text: string,
  predictions: Array<{ word: string; entity: string; score: number }>,
): {
  alignedThrough: number;
  entities: Array<{ label: string; text: string; start: number; end: number; score: number }>;
};
export function createBrowserNer(options?: BrowserNerOptions): BrowserNer;
