import { StackIntro } from "./StackIntro";
import { TextAnalysisStudio } from "./TextAnalysisStudio";

export default function Home() {
  return (
    <main>
      <StackIntro />
      <div className="mx-auto max-w-6xl space-y-10 px-5 py-10 sm:px-8 sm:py-14">
        <section aria-labelledby="browser-demo-heading">
          <div className="mb-6 max-w-3xl">
            <p className="text-sm font-semibold uppercase tracking-[0.12em] text-accent">Browser analysis workbench</p>
            <h2 id="browser-demo-heading" className="mt-2 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
              Text analysis studio
            </h2>
            <p className="mt-3 text-base leading-7 text-muted">
              Paste text directly or upload a document. Text-bearing files are parsed locally; scanned PDFs and images can be OCRed before the extracted text is handed to the same deterministic text-analysis surface used elsewhere in the workspace.
            </p>
          </div>
          <TextAnalysisStudio />
        </section>

        <section className="border-t border-line pt-8" aria-labelledby="architecture-heading">
          <h2 id="architecture-heading" className="text-lg font-semibold text-ink">Ownership boundary</h2>
          <ul className="mt-4 grid gap-3 text-sm leading-6 text-muted md:grid-cols-3">
            <li className="border-l-2 border-line pl-4">The browser owns file selection, PDF text extraction, rendering, and OCR adaptation.</li>
            <li className="border-l-2 border-line pl-4">Rust text-analysis owns summaries, lexical evidence, linguistic analysis, embeddings, and semantic structure.</li>
            <li className="border-l-2 border-line pl-4">GitHub Pages remains a static host; analysis does not depend on an nlp-stack API server.</li>
          </ul>
        </section>
      </div>
    </main>
  );
}
