import { createTextResultTabs, PackageSurfaceWorkbench, type PackageAppConfig } from "@moritzbrantner/nlp-app-ui/package-surface";
import * as wasm from "@moritzbrantner/text-transcripts-wasm";

const packageAppConfig: PackageAppConfig = {
  library: "text-transcripts",
  title: "Text Transcripts",
  description: "Transcript parsing and ASR command adapters for video-analysis.",
  domain: "text",
  wasm: {
    init: wasm.init,
    packageSurface: wasm.packageSurface,
    runOperation: wasm.runOperation,
  },
  server: {
    scopedRoute: "/api/rust/packages/text-transcripts",
    standaloneRoute: "",
  },
  defaultOperation: "transcripts.parse",
  featuredOperations: ["transcripts.parse", "transcripts.normalize", "transcripts.importWhisperX", "transcripts.formatSrt", "transcripts.formatWebVtt", "transcripts.toTextSegments", "describe"],
  operationGroups: [
    {
      id: "workflow",
      label: "Workflow",
      description: "Run transcript parsing, normalization, caption formatting, and text segment conversion workflows.",
      operations: ["transcripts.parse", "transcripts.normalize", "transcripts.importWhisperX", "transcripts.formatSrt", "transcripts.formatWebVtt", "transcripts.toTextSegments"],
    },
    {
      id: "debug",
      label: "Debug",
      description: "Inspect package metadata and operation support.",
      operations: ["describe"],
    },
  ],
  presets: [
    {
      id: "parse-srt",
      label: "Parse SRT",
      operation: "transcripts.parse",
      description: "Parse SRT content into the transcript contract.",
      input: {
        format: "srt",
        content: "1\n00:00:01,000 --> 00:00:02,200\nAlice presented the tokenizer roadmap.\n\n2\n00:00:02,500 --> 00:00:04,000\nBob reviewed transcript retrieval evidence.\n",
      },
    },
    {
      id: "parse-webvtt",
      label: "Parse WebVTT",
      operation: "transcripts.parse",
      description: "Parse WebVTT captions into the transcript contract.",
      input: {
        format: "webVtt",
        content: "WEBVTT\n\n00:00:01.000 --> 00:00:02.200\nAlice presented the tokenizer roadmap.\n\n00:00:02.500 --> 00:00:04.000\nBob reviewed transcript retrieval evidence.\n",
      },
    },
    {
      id: "parse-plain",
      label: "Parse plain lines",
      operation: "transcripts.parse",
      description: "Parse plain transcript lines into segment records.",
      input: { format: "plain", content: "Alice presented the tokenizer roadmap.\nBob reviewed transcript retrieval evidence.\n" },
    },
    {
      id: "normalize",
      label: "Normalize transcript",
      operation: "transcripts.normalize",
      description: "Normalize segment text, final flags, and joined transcript text.",
      input: {
        segments: [
          { index: 0, startSeconds: 1.0, endSeconds: 2.2, text: "  Alice   presented the tokenizer roadmap. ", isFinal: true },
          { index: 1, startSeconds: 2.5, endSeconds: 4.0, text: " Bob reviewed transcript retrieval evidence. ", isFinal: true },
        ],
      },
    },
    {
      id: "import-whisperx",
      label: "Import WhisperX",
      operation: "transcripts.importWhisperX",
      description: "Import existing WhisperX JSON into the transcript contract.",
      input: {
        content: "{\"segments\":[{\"start\":0.0,\"end\":1.0,\"text\":\" Alice presented the roadmap. \",\"words\":[{\"word\":\"Alice\",\"start\":0.0,\"end\":0.3,\"score\":0.94,\"speaker\":\"SPEAKER_00\"},{\"word\":\"presented\",\"start\":0.32,\"end\":0.72,\"score\":0.9,\"speaker\":\"SPEAKER_00\"}]}]}",
      },
    },
    {
      id: "format-srt",
      label: "Format SRT",
      operation: "transcripts.formatSrt",
      description: "Format a transcript contract as SRT.",
      input: {
        segments: [
          { index: 0, startSeconds: 1.0, endSeconds: 2.2, text: "Alice presented the tokenizer roadmap.", isFinal: true },
          { index: 1, startSeconds: 2.5, endSeconds: 4.0, text: "Bob reviewed transcript retrieval evidence.", isFinal: true },
        ],
      },
    },
    {
      id: "format-webvtt",
      label: "Format WebVTT",
      operation: "transcripts.formatWebVtt",
      description: "Format a transcript contract as WebVTT.",
      input: {
        segments: [
          { index: 0, startSeconds: 1.0, endSeconds: 2.2, text: "Alice presented the tokenizer roadmap.", isFinal: true },
          { index: 1, startSeconds: 2.5, endSeconds: 4.0, text: "Bob reviewed transcript retrieval evidence.", isFinal: true },
        ],
      },
    },
    {
      id: "to-text-segments",
      label: "To text segments",
      operation: "transcripts.toTextSegments",
      description: "Convert timed transcript segments into shared text segment and document contracts.",
      input: {
        streamId: "transcript-1",
        segments: [
          { index: 0, startSeconds: 1.0, endSeconds: 2.2, text: "Alice presented the tokenizer roadmap.", language: "en", speaker: "A", confidence: 0.92, isFinal: true },
          { index: 1, startSeconds: 2.5, endSeconds: 4.0, text: "Bob reviewed transcript retrieval evidence.", language: "en", speaker: "B", confidence: 0.88, isFinal: true },
        ],
      },
    },
  ],
  benchmarkScenarios: [
    {
      id: "parse-srt",
      label: "Parse SRT",
      operation: "transcripts.parse",
      input: { format: "srt", content: "1\n00:00:01,000 --> 00:00:02,000\nHello benchmark.\n\n2\n00:00:02,000 --> 00:00:03,000\nRust transcript parsing.\n" },
      iterations: 120,
      warmupIterations: 5,
      outputCountPath: ["segments"],
    },
    {
      id: "normalize",
      label: "Normalize",
      operation: "transcripts.normalize",
      input: { segments: [{ index: 0, startSeconds: 1.0, endSeconds: 2.0, text: "  Hello   benchmark.  ", isFinal: true }] },
      iterations: 120,
      warmupIterations: 5,
      outputCountPath: ["segments"],
    },
    {
      id: "import-whisperx",
      label: "Import WhisperX",
      operation: "transcripts.importWhisperX",
      input: { content: "{\"segments\":[{\"start\":0.0,\"end\":1.0,\"text\":\"Hello benchmark.\",\"words\":[{\"word\":\"Hello\",\"start\":0.0,\"end\":0.5,\"score\":0.9}]}]}" },
      iterations: 120,
      warmupIterations: 5,
      outputCountPath: ["segments"],
    },
    {
      id: "format-srt",
      label: "Format SRT",
      operation: "transcripts.formatSrt",
      input: { segments: [{ index: 0, startSeconds: 1.0, endSeconds: 2.0, text: "Hello.", isFinal: true }] },
      iterations: 120,
      warmupIterations: 5,
      outputCountPath: ["content"],
    },
    {
      id: "format-webvtt",
      label: "Format WebVTT",
      operation: "transcripts.formatWebVtt",
      input: { segments: [{ index: 0, startSeconds: 1.0, endSeconds: 2.0, text: "Hello.", isFinal: true }] },
      iterations: 120,
      warmupIterations: 5,
      outputCountPath: ["webVtt"],
    },
    {
      id: "to-text-segments",
      label: "To Text Segments",
      operation: "transcripts.toTextSegments",
      input: { streamId: "transcript-1", segments: [{ index: 0, startSeconds: 1.0, endSeconds: 2.0, text: "Hello.", language: "en", speaker: "A", confidence: 0.9, isFinal: true }] },
      iterations: 120,
      warmupIterations: 5,
      outputCountPath: ["segments"],
    },
  ],
  resultTabs: createTextResultTabs({
    library: "text-transcripts",
    primaryOperations: {
      "transcripts.parse": {
        title: "Transcript parsing",
        summaryFields: ["segmentCount", "hasText"],
        listFields: ["segments", "words"],
        objectFields: ["metadata", "result"],
        explanation: () => "The parser converted SRT, WebVTT, Whisper JSON, or plain lines into the shared transcript contract and normalized joined text when available.",
      },
      "transcripts.normalize": {
        title: "Transcript normalization",
        summaryFields: ["segmentCount", "hasText"],
        listFields: ["segments", "words"],
        objectFields: ["metadata", "result"],
        explanation: () => "The normalization pass cleaned segment text, preserved timing, and rebuilt contract-level transcript text.",
      },
      "transcripts.importWhisperX": {
        title: "WhisperX import",
        summaryFields: ["segmentCount", "hasText"],
        listFields: ["segments", "words"],
        objectFields: ["metadata", "result"],
        explanation: () => "The importer converted WhisperX JSON into the shared transcript contract while preserving word timings and speaker labels.",
      },
      "transcripts.formatSrt": {
        title: "SRT formatting",
        summaryFields: ["bytes"],
        objectFields: ["result"],
        explanation: () => "The formatter normalized the transcript contract and emitted SRT caption text with stable timing.",
      },
      "transcripts.formatWebVtt": {
        title: "WebVTT formatting",
        summaryFields: ["bytes"],
        objectFields: ["result"],
        explanation: () => "The formatter normalized the transcript contract and emitted WebVTT caption text with stable timing.",
      },
      "transcripts.toTextSegments": {
        title: "Text segment conversion",
        summaryFields: ["segmentCount", "documentCount", "streamId"],
        listFields: ["segments", "documents"],
        objectFields: ["result"],
        explanation: () => "The converter mapped transcript timing, language, speaker, and confidence fields into shared text segment contracts and document records.",
      },
    },
  }),
};

export function App() {
  return <PackageSurfaceWorkbench config={packageAppConfig} />;
}
