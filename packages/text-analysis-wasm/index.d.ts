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

export interface AnalyzeDocumentInput {
  id?: string;
  text: string;
  profile?: "deterministic" | "modelBacked" | "model-backed";
  keywordLimit?: number;
  summarySentences?: number;
  ngramSizes?: number[];
  shingleSizes?: number[];
  linguistics?: { mode: string; bundleDir?: string; autoDownload?: boolean; downloadProgress?: boolean };
  embedding?: { mode: string; dimensions?: number; useIdf?: boolean; bundleDir?: string };
}

export interface AnalyzeCorpusInput {
  documents: Array<{ id?: string; text: string }>;
  query?: string;
  topK?: number;
  keywordLimit?: number;
  tfidfTermsPerDocument?: number;
  includeNearDuplicates?: boolean;
  includeSemanticNeighbors?: boolean;
  embedding?: { mode: string; dimensions?: number; useIdf?: boolean; bundleDir?: string };
}

export interface SimilarityInput {
  left: string;
  right: string;
  n?: number;
  mode?: "token" | "character" | "char";
}

export type DocumentAnalysisReport = Record<string, unknown>;
export type CorpusAnalysisReport = Record<string, unknown>;
export type SimilarityReport = Record<string, unknown>;

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
export function analyzeDocument(input: AnalyzeDocumentInput): Promise<DocumentAnalysisReport>;
export function analyzeCorpus(input: AnalyzeCorpusInput): Promise<CorpusAnalysisReport>;
export function compareTexts(input: SimilarityInput): Promise<SimilarityReport>;
