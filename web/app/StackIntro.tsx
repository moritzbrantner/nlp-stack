export function StackIntro() {
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto max-w-6xl px-5 py-12 sm:px-8 sm:py-16">
        <p className="text-sm font-semibold uppercase tracking-[0.16em] text-accent">nlp-stack</p>
        <h1 className="mt-3 max-w-4xl text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
          Analyze text and documents locally with Rust NLP.
        </h1>
        <p className="mt-5 max-w-3xl text-base leading-7 text-muted sm:text-lg">
          Paste text, upload a document, or OCR an image and inspect summaries, keywords, entities, linguistic evidence, and semantic structure. Browser ingestion stays an adapter; the analysis itself runs through the repository&apos;s Rust crates compiled to WebAssembly.
        </p>
        <div className="mt-7 flex flex-wrap gap-x-5 gap-y-3 text-sm font-medium">
          <a className="text-accent underline underline-offset-4" href="https://github.com/moritzbrantner/nlp-stack">
            View source
          </a>
          <span className="text-muted">Static GitHub Pages · local file processing · Rust/Wasm analysis</span>
        </div>
      </div>
    </header>
  );
}
