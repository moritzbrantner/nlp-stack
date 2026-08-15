import path from "node:path";
import { fileURLToPath } from "node:url";

const playwright = await import("playwright").catch(() => null);
if (!playwright) {
  console.error("Playwright is not installed. Run `bun install` before `bun run text-wasm:bench:all`.");
  process.exit(0);
}
const { chromium } = playwright;

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const hostname = "127.0.0.1";

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".wasm", "application/wasm"],
  [".json", "application/json; charset=utf-8"],
]);

const scenarios = [
  ["text-core", "text.tokenize", { text: "Rust text analysis ".repeat(128), includeStats: true }, 60, ["tokens"]],
  ["text-lexical", "lexical.keywords", { text: "Rust text analysis supports lexical search. ".repeat(64), maxTerms: 16 }, 50, ["keywords"]],
  ["text-embeddings", "embeddings.embed", { texts: ["rust text analysis", "semantic transcript retrieval"], dimensions: 128 }, 40, ["embeddings"]],
  ["text-retrieval", "retrieval.search", { documents: [{ id: "a", body: "rust text retrieval" }, { id: "b", body: "scene reports" }], query: "text", mode: "hybrid" }, 40, ["results"]],
  ["text-analysis", "analysis.document", { id: "bench", text: "Alice presented tokenizer roadmap. ".repeat(16), profile: "deterministic", embedding: { mode: "hashed", dimensions: 64 } }, 20, ["summary"]],
  ["text-linguistics", "linguistics.analyze", { text: "Alice presented the tokenizer roadmap in Berlin.", profile: "fast" }, 30, ["entities"]],
  ["text-model-runtime", "runtime.softmax", { logits: [0.1, 0.3, 1.2, -0.7, 2.4] }, 100, ["probabilities"]],
  ["text-classification", "classification.classify", { text: "rust is reliable", labels: ["positive", "negative"], model: { fallbackPolicy: "lexical_fallback" } }, 50, ["predictions"]],
  ["text-question-answering", "qa.answer", { question: "What is reliable?", context: "Rust is reliable.", importedPredictions: [{ text: "Rust", score: 0.9 }] }, 50, ["answers"]],
  ["text-generation", "generation.markovGenerate", { trainingTexts: ["rust text analysis supports crates"], order: 2, maxTokens: 8 }, 40, ["tokens"]],
  ["text-generation-linguistics", "generationLinguistics.analysisTerms", { text: "Scene transitions follow transcript cues." }, 30, ["terms"]],
  ["text-transcripts", "transcripts.parse", { format: "srt", content: "1\n00:00:01,000 --> 00:00:02,000\nHello.\n" }, 60, ["segments"]],
].map(([library, operation, input, iterations, outputCountPath]) => ({
  library,
  packageDir: `packages/${library}-wasm`,
  operation,
  input,
  iterations,
  outputCountPath,
}));

function responseForPath(requestPath) {
  const relativePath = requestPath === "/" ? "packages/text-benchmarks/bench/runner.html" : decodeURIComponent(requestPath.slice(1));
  const filePath = path.resolve(repoRoot, relativePath);

  if (!filePath.startsWith(`${repoRoot}${path.sep}`)) {
    return new Response("Not found", { status: 404 });
  }

  return new Response(Bun.file(filePath), {
    headers: {
      "content-type": contentTypes.get(path.extname(filePath)) ?? "application/octet-stream",
    },
  });
}

function printResults(results) {
  console.log(`Browser: ${results.userAgent}`);
  console.log("package,operation,iterations,total_ms,average_ms,ops_per_second,output_count,status");
  for (const entry of results.entries) {
    console.log(
      [
        entry.library,
        entry.operation,
        entry.iterations,
        entry.totalMs?.toFixed?.(3) ?? "",
        entry.averageMs?.toFixed?.(3) ?? "",
        entry.opsPerSecond?.toFixed?.(2) ?? "",
        entry.outputCount ?? "",
        entry.status,
      ].join(","),
    );
    if (entry.error) {
      console.error(`${entry.library}: ${entry.error}`);
    }
  }
}

const server = Bun.serve({
  hostname,
  port: 0,
  fetch(request) {
    return responseForPath(new URL(request.url).pathname);
  },
});

let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage();
  page.setDefaultTimeout(60_000);
  page.on("pageerror", (error) => console.error(error));
  await page.goto(`http://${hostname}:${server.port}/packages/text-benchmarks/bench/runner.html`);
  const results = await page.evaluate(async (inputScenarios) => globalThis.runTextWasmBench(inputScenarios), scenarios);
  printResults(results);
} finally {
  await browser?.close();
  server.stop(true);
}
