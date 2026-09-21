import { expect, test } from "playwright/test";

test("keeps single-document semantics separate from multi-document corpus themes", async ({ page }) => {
  // The test server supplies the model fixture; workers and Rust/WASM remain real.
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: /Analyze text and documents locally with Rust NLP/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Try an example" })).toBeVisible();
  await expect(page.getByLabel("Text to analyze")).toBeVisible();
  await expect(page.getByLabel("Upload documents")).toHaveAttribute("multiple", "");

  const dialogue = page.getByRole("button", { name: /Meeting dialogue/ });
  await expect(dialogue).toBeVisible();
  await dialogue.click();
  await expect(page.getByLabel("Text to analyze")).toHaveValue(/Maya:/);

  const analyze = page.getByRole("button", { name: "Analyze text" });
  await expect(analyze).toBeEnabled();
  await analyze.click();

  await expect(page.getByRole("heading", { name: "Semantic map" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Source: Meeting dialogue/i)).toBeVisible();

  await page.getByRole("tab", { name: "Corpus themes" }).click();
  await expect(page.getByRole("heading", { name: "Corpus themes" })).toBeVisible();
  await expect(page.getByText(/require at least two supplied documents/i)).toBeVisible();

  await page.getByRole("tab", { name: "Word profile" }).click();
  await expect(page.getByRole("heading", { name: "Word profile" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ranked terms" })).toBeVisible();

  await page.getByLabel("Upload documents").setInputFiles([
    {
      name: "search-a.txt",
      mimeType: "text/plain",
      buffer: Buffer.from(
        "Semantic search improves retrieval. Vector indexes support semantic search. Tomatoes grow in soil.",
      ),
    },
    {
      name: "search-b.txt",
      mimeType: "text/plain",
      buffer: Buffer.from(
        "Semantic search improves retrieval. Vector indexes support semantic search. Roses grow in soil.",
      ),
    },
  ]);

  await expect(page.getByRole("heading", { name: "Corpus themes" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/Corpus: 2 sources/i)).toBeVisible();
  await expect(page.getByText(/Embedding evidence: fixture\/browser-semantic-model/)).toBeVisible();
  await expect(page.getByText(/hashed TF-IDF backend/i)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Theme evidence" })).toBeVisible();
  await expect(page.getByText(/semantic.*retrieval|retrieval.*semantic/i).first()).toBeVisible();
  await expect(page.getByText(/Analysis ready. Corpus themes use 2 supplied sources/i)).toBeVisible();
});

test("cancels a busy worker while the page stays responsive and can analyze again", async ({ page }) => {
  // A synchronous loop cannot handle a cancellation message; termination must stop it.
  await page.route("**/nlp-analysis-worker.js", (route) => route.fulfill({
    contentType: "application/javascript",
    body: `self.addEventListener("message", () => { console.log("analysis-fixture-running"); while (true) {} });`,
  }), { times: 1 });
  await page.goto("/");
  const started = page.waitForEvent("console", { predicate: (message) => message.text() === "analysis-fixture-running" });
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
  await started;
  // The baseline studio has an independent job; it can finish while this worker is busy.
  await page.getByLabel("Text", { exact: true }).fill("Cats sleep.");
  await page.getByRole("button", { name: "Add text to corpus", exact: true }).click();
  await page.getByRole("button", { name: "Build semantic baseline", exact: true }).click();
  await expect(page.getByText(/Baseline ready with 1 source/)).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "Cancel analysis", exact: true }).click();
  await expect(page.getByRole("button", { name: "Export JSON", exact: true })).toBeEnabled();
  await expect(page.getByText("Analysis cancelled.", { exact: true })).toBeVisible();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
  await expect(page.getByText(/Analysis ready. Single-document/)).toBeVisible({ timeout: 20_000 });
});

test("rejects oversized analysis with an actionable message and allows a smaller input", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Text to analyze").fill("Cats sleep. ".repeat(513));
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("512 sentences");
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Choose a shorter passage");
  await page.getByLabel("Text to analyze").fill("Cats sleep.");
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();
  await expect(page.getByText(/Analysis ready. Single-document/)).toBeVisible({ timeout: 20_000 });
});

test("cancels baseline analysis without enabling export and can rebuild", async ({ page }) => {
  await page.route("**/nlp-analysis-worker.js", (route) => route.fulfill({
    contentType: "application/javascript",
    body: `self.addEventListener("message", () => { console.log("baseline-fixture-running"); while (true) {} });`,
  }), { times: 1 });
  await page.goto("/");
  await page.getByLabel("Text", { exact: true }).fill("Cats sleep.");
  await page.getByRole("button", { name: "Add text to corpus", exact: true }).click();
  const started = page.waitForEvent("console", { predicate: (message) => message.text() === "baseline-fixture-running" });
  await page.getByRole("button", { name: "Build semantic baseline", exact: true }).click();
  await started;
  await page.getByRole("button", { name: "Cancel baseline analysis", exact: true }).click();
  await expect(page.getByText("Baseline analysis cancelled.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Export JSON", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Build semantic baseline", exact: true }).click();
  await expect(page.getByText(/Baseline ready with 1 source/)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("button", { name: "Export JSON", exact: true })).toBeEnabled();
});
