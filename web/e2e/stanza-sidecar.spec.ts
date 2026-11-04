import { readFileSync } from "node:fs";

import { expect, test } from "playwright/test";

// The opt-in local Stanza sidecar (#190). The browser posts the Rust/Wasm-built
// request to the sidecar URL; this test answers it with the committed recording of
// the real sidecar, so Rust/Wasm, workers and routing stay real and CI stays offline.

const fixtures = new URL("../../crates/text/text-linguistics/tests/fixtures/stanza/", import.meta.url);
const sample = readFileSync(new URL("sample-en-de-es.txt", fixtures), "utf8").trim();
const recorded = readFileSync(new URL("sample-en-de-es.json", fixtures), "utf8");
const sidecar = "http://127.0.0.1:8765/v1/analyze";

test("shows real en/de/es syntax from the local Stanza sidecar", async ({ page }) => {
  const posted: unknown[] = [];
  await page.route(sidecar, async (route) => {
    posted.push(route.request().postDataJSON());
    await route.fulfill({ status: 200, contentType: "application/json", body: recorded });
  });
  await page.goto("/");
  await page.getByLabel("Text to analyze").fill(sample);
  await page.getByLabel(/Use the Stanza sidecar/).check();
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
  await page.getByRole("tab", { name: "Sentences" }).click();

  const languages = [
    ["Sentence 1: English (en)", "en"],
    ["Sentence 2: German (de)", "de"],
    ["Sentence 3: Spanish (es)", "es"],
  ] as const;
  for (const [name, language] of languages) {
    const card = page.getByRole("article", { name, exact: true });
    await expect(card).toBeVisible({ timeout: 30_000 });
    for (const operation of ["Lemma and morphology", "Part of speech", "Dependency syntax"]) {
      const row = card.getByRole("row", { name: new RegExp(`^${operation}`) });
      await expect(row.getByRole("cell").first()).toHaveText("Supported");
      await expect(row).toContainText(`stanza-1.15.0:${language}/combined_charlm`);
    }
    await expect(card).toContainText(`stanfordnlp/stanza-${language}@`);
  }
  expect(posted).toHaveLength(1);
  expect((posted[0] as { sentences: { language: string }[] }).sentences.map((item) => item.language)).toEqual(["en", "de", "es"]);

  const german = page.getByRole("article", { name: "Sentence 2: German (de)", exact: true }).getByRole("table", { name: "Syntactic words" });
  const zu = german.getByRole("row", { name: /^zu\s*\(in zum\)/ });
  await expect(zu).toContainText("ADP");
  await expect(zu).toContainText("case → Bahnhof");
  const dem = german.getByRole("row", { name: /^dem\s*\(in zum\)/ });
  await expect(dem).toContainText("Case=Dat");
  const spanish = page.getByRole("article", { name: "Sentence 3: Spanish (es)", exact: true }).getByRole("table", { name: "Syntactic words" });
  await expect(spanish.getByRole("row", { name: /^niño/ })).toContainText("nsubj → volvió");
});

test("stops with the reason when the Stanza sidecar is not running", async ({ page }) => {
  await page.route(sidecar, (route) => route.abort("connectionrefused"));
  await page.goto("/");
  await page.getByLabel("Text to analyze").fill(sample);
  await page.getByLabel(/Use the Stanza sidecar/).check();
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
  await expect(page.getByRole("alert").filter({ hasText: /Stanza sidecar/ })).toContainText(
    /Stanza sidecar at http:\/\/127\.0\.0\.1:8765 is not reachable/,
    { timeout: 30_000 },
  );
  await expect(page.getByRole("heading", { name: "Analysis", exact: true })).toHaveCount(0);
});
