import { expect, test } from "playwright/test";

// The en/de/es journey over the seven core operations (#116, #117). Rust/Wasm, workers
// and per-sentence routing are real; only the NER model is the E2E server's fixture.

type Status = "Supported" | "Partial" | "Unsupported" | "Unknown" | "Rust only";

const operations = [
  "Language detection",
  "Sentence segmentation",
  "Tokenization",
  "Named entities",
  "Lemma and morphology",
  "Part of speech",
  "Dependency syntax",
] as const;

const sentences: {
  name: string;
  text: string;
  bytes: string;
  statuses: Status[];
  evidence: Partial<Record<(typeof operations)[number], RegExp>>;
}[] = [
  {
    name: "Sentence 1: English (en)",
    text: "Sarah says John is late.",
    bytes: "0–24",
    statuses: ["Supported", "Supported", "Supported", "Supported", "Partial", "Partial", "Partial"],
    evidence: {
      "Language detection": /detected \(en 0\.\d\d/,
      Tokenization: /5 tokens/,
      "Named entities": /distilbert-base-multilingual-cased-ner-hrl@c2a4dbf5 \(browser\) · 2 entities/,
      "Part of speech": /heuristic:english-rules · English rules; unscored/,
    },
  },
  {
    name: "Sentence 2: German (de)",
    text: "Er glaubt, dass John nicht kommen wird.",
    bytes: "25–64",
    statuses: ["Supported", "Supported", "Supported", "Supported", "Rust only", "Rust only", "Rust only"],
    evidence: {
      "Language detection": /detected \(de 0\.\d\d/,
      "Named entities": /1 entity/,
      "Lemma and morphology": /German UD model .* runs only in the native Rust library/,
      "Dependency syntax": /not available in the browser/,
    },
  },
  {
    name: "Sentence 3: Spanish (es)",
    text: "María explicó que Ana llegó tarde.",
    bytes: "65–102",
    statuses: ["Supported", "Supported", "Supported", "Supported", "Rust only", "Rust only", "Rust only"],
    evidence: {
      "Language detection": /detected \(es 0\.\d\d/,
      "Named entities": /2 entities/,
      "Part of speech": /Spanish UD model/,
    },
  },
  {
    name: "Sentence 4: Undecided",
    text: "Thanks.",
    bytes: "103–110",
    statuses: ["Unknown", "Supported", "Supported", "Unknown", "Unknown", "Unknown", "Unknown"],
    evidence: {
      "Language detection": /too few words to decide a language/,
      "Named entities": /language undecided, so the model was not run/,
    },
  },
  {
    name: "Sentence 5: French (fr)",
    text: "Nous partirons demain matin pour Lyon.",
    bytes: "111–149",
    statuses: ["Supported", "Supported", "Supported", "Unsupported", "Unsupported", "Unsupported", "Unsupported"],
    evidence: {
      "Named entities": /supports en, de, es, not fr/,
      "Part of speech": /no model for `fr`/,
    },
  },
];

test("analyzes each en/de/es sentence in its own language and reports unsupported operations", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Sentence languages/ }).click();
  await expect(page.getByLabel("Text to analyze")).toHaveValue(/María explicó que Ana llegó tarde\./);
  await page.getByRole("button", { name: "Analyze text", exact: true }).click();

  await expect(page.getByRole("heading", { name: "Sentences", exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("article")).toHaveCount(sentences.length);
  for (const sentence of sentences) {
    const card = page.getByRole("article", { name: sentence.name, exact: true });
    await expect(card).toBeVisible();
    await expect(card.getByRole("blockquote")).toHaveText(sentence.text);
    await expect(card).toContainText(`UTF-8 bytes ${sentence.bytes}`);
    for (const [index, operation] of operations.entries()) {
      const row = card.getByRole("row", { name: new RegExp(`^${operation}`) });
      await expect(row.getByRole("cell").first(), `${sentence.name} · ${operation}`).toHaveText(sentence.statuses[index]!);
      const evidence = sentence.evidence[operation];
      if (evidence) await expect(row).toContainText(evidence);
    }
  }

  // Entities come only from sentences whose language the model supports: French
  // "Lyon" stays unlabelled although the fixture model would tag it.
  await page.getByRole("tab", { name: "Entities" }).click();
  await expect(page.getByText(/2 of 5 sentences were not labelled/)).toBeVisible();
  for (const name of ["Sarah", "John", "María", "Ana"]) {
    await expect(page.getByRole("cell", { name, exact: true }).first()).toBeVisible();
  }
  await expect(page.getByRole("cell", { name: "Lyon", exact: true })).toHaveCount(0);
});
