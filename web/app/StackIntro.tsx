export function StackIntro() {
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto max-w-6xl px-5 py-12 sm:px-8 sm:py-16">
        <p className="text-sm font-semibold uppercase tracking-[0.16em] text-accent">nlp-stack</p>
        <h1 className="mt-3 max-w-4xl text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
          Rust NLP package surfaces, running in the browser.
        </h1>
        <p className="mt-5 max-w-3xl text-base leading-7 text-muted sm:text-lg">
          The same deterministic Rust operations used by the workspace compile to WebAssembly and run inside a static Next.js site. The page stays a thin React host; NLP behavior remains owned by the Rust crates.
        </p>
        <div className="mt-7 flex flex-wrap gap-x-5 gap-y-3 text-sm font-medium">
          <a className="text-accent underline underline-offset-4" href="https://github.com/moritzbrantner/nlp-stack">
            View source
          </a>
          <span className="text-muted">Rust → wasm-bindgen → React → static Next.js</span>
        </div>
      </div>
    </header>
  );
}
