import { StackIntro } from "./StackIntro";
import { TextCoreDemo } from "./TextCoreDemo";

export default function Home() {
  return (
    <main>
      <StackIntro />
      <div className="mx-auto max-w-6xl space-y-10 px-5 py-10 sm:px-8 sm:py-14">
        <section aria-labelledby="browser-demo-heading">
          <div className="mb-5 max-w-3xl">
            <p className="text-sm font-semibold uppercase tracking-[0.12em] text-accent">Browser slice</p>
            <h2 id="browser-demo-heading" className="mt-2 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
              Text Core
            </h2>
            <p className="mt-3 text-base leading-7 text-muted">
              Choose an operation, edit its structured input, and execute the repository&apos;s existing text-core Wasm adapter locally. No API server is involved.
            </p>
          </div>
          <TextCoreDemo />
        </section>

        <section className="border-t border-line pt-8" aria-labelledby="architecture-heading">
          <h2 id="architecture-heading" className="text-lg font-semibold text-ink">What this page proves</h2>
          <ul className="mt-4 grid gap-3 text-sm leading-6 text-muted md:grid-cols-3">
            <li className="border-l-2 border-line pl-4">Rust owns package operations and request/response semantics.</li>
            <li className="border-l-2 border-line pl-4">wasm-bindgen exposes the same package surface to the browser.</li>
            <li className="border-l-2 border-line pl-4">Next exports only static assets suitable for GitHub Pages.</li>
          </ul>
        </section>
      </div>
    </main>
  );
}
