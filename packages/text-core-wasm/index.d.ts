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

export interface TextSpanResult {
  end: number;
  start: number;
  text: string;
}

export interface TextTokenResult extends TextSpanResult {
  kind: string;
  normalized?: string;
}

export interface SegmentedTextDocument {
  paragraphs: TextSpanResult[];
  sentences: TextSpanResult[];
  tokens: TextTokenResult[];
}

export interface TextDocumentAnalysis extends SegmentedTextDocument {
  scriptProfile: {
    digits: number;
    dominantScript: string | null;
    isMixed: boolean;
    other: number;
    punctuation: number;
    scripts: Record<string, number>;
    whitespace: number;
  };
  stats: {
    averageCharsPerWord: number;
    averageWordsPerSentence: number;
    basic: unknown;
    paragraphs: number;
    sentences: number;
    tokens: number;
    uniqueTokens: number;
  };
}

export function init(input?: unknown): Promise<unknown>;
export function packageSurface(): PackageSurface;
export function runOperation(request: SurfaceRequest): SurfaceResponse;
export function extractWordTexts(text: string): string[];
export function splitSentences(text: string): string[];
export function segmentTextDocument(
  text: string,
  keepApostrophes?: boolean,
  includePunctuation?: boolean,
  includeTokens?: boolean,
): SegmentedTextDocument;
export function analyzeTextDocument(
  text: string,
  options?: {
    includePunctuation?: boolean;
    includeTokens?: boolean;
  },
): TextDocumentAnalysis;
export default init;
