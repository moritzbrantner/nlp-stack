export interface SurfaceRequest {
  operation: string;
  input: unknown;
}

export interface SurfaceOperation {
  id: string;
  name: string;
  description?: string;
  inputSchema: unknown;
  outputSchema: unknown;
  exampleRequest: unknown;
  wasmSupported: boolean;
  serverSupported: boolean;
}

export interface PackageSurface {
  library: string;
  version: string;
  operations: SurfaceOperation[];
  capabilities: unknown;
}

export interface SurfaceResponse {
  operation: string;
  value: unknown;
  diagnostics: unknown[];
  artifacts: unknown[];
}

export type IndexSearchMode = "lexical" | "semantic" | "hybrid";
export type ChunkingStrategy = "tokenWindow" | "sentence" | "paragraph";

export interface IndexDocument {
  id: string;
  title?: string;
  body: string;
  language?: string;
  metadata?: {
    attributes?: Record<string, string>;
    source?: unknown;
    provenance?: unknown[];
    annotations?: unknown[];
  };
  analysisAttachments?: unknown[];
  semanticFacets?: unknown[];
}

export interface IndexBuildOptions {
  chunkingStrategy?: ChunkingStrategy;
  chunkTokens?: number;
  chunkOverlapTokens?: number;
  storeRawText?: boolean;
  commit?: boolean;
  processing?: unknown;
}

export interface IndexQuery {
  text: string;
  /** Required. Semantic and hybrid need caller-computed `importedEmbeddings` (the WASM surface cannot load a model). */
  mode: IndexSearchMode;
  topK?: number;
  candidateLimit?: number;
  filter?: unknown;
  semanticWeight?: number;
  lexicalWeight?: number;
  requiredPhrases?: string[];
  explain?: boolean;
}

export type EmbeddingInputKind = "query" | "passage" | "symmetric";

export interface ImportedEmbedding {
  /** The exact text (chunk text or query text) the vector was computed for, without model prefixes. */
  text: string;
  vector: number[];
  /** Role the vector was computed for; omitted serves every role. */
  kind?: EmbeddingInputKind;
}

export interface ImportedEmbeddingModel {
  name: string;
  dimensions?: number;
  maxTokens?: number;
  revision?: string;
  pooling?: "cls" | "mean" | "last_token";
}

export interface IndexSearchRequest {
  backend?: "memory" | "sqlite" | string;
  path?: string;
  commit?: boolean;
  documents: IndexDocument[];
  query: IndexQuery;
  options?: IndexBuildOptions;
  /** Provenance of `importedEmbeddings`: model id, revision, pooling. */
  embeddingModel?: ImportedEmbeddingModel;
  /**
   * Caller-computed vectors for semantic and hybrid search: a `passage` vector for
   * every chunk text and a `query` vector for `query.text`. Lexical search ignores them.
   */
  importedEmbeddings?: ImportedEmbedding[];
}

export interface IndexChunk {
  id: string;
  documentId: string;
  ordinal: number;
  text: string;
  byteStart: number;
  byteEnd: number;
  tokenStart?: number;
  tokenEnd?: number;
  metadata?: Record<string, string>;
  source?: unknown;
  provenance?: unknown[];
  annotations?: unknown[];
  semanticFacets?: unknown[];
}

export interface IndexScoreBreakdown {
  semanticScore: number;
  lexicalScore: number;
  normalizedSemanticScore: number;
  normalizedLexicalScore: number;
  semanticWeight: number;
  lexicalWeight: number;
  explanation?: string;
}

export interface IndexSearchResult {
  chunkId: string;
  documentId: string;
  score: number;
  snippet: string;
  matchedPhrases: string[];
  chunk: IndexChunk;
  scoreBreakdown: IndexScoreBreakdown;
}

export interface IndexSearchSurfaceValue {
  operation: "index.search";
  title: string;
  message: string;
  summary: unknown;
  result: {
    backend: string;
    results: IndexSearchResult[];
  };
}

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
