import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StackIntro } from "./StackIntro";

describe("StackIntro", () => {
  it("explains the local Rust/Wasm analysis boundary and links to the repository", () => {
    render(<StackIntro />);

    expect(
      screen.getByRole("heading", { level: 1, name: /Analyze text and documents locally with Rust NLP/i }),
    ).toBeTruthy();
    expect(screen.getByText(/Browser ingestion stays an adapter/i)).toBeTruthy();
    expect(screen.getByText(/Static GitHub Pages · local file processing · Rust\/Wasm analysis/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "View source" }).getAttribute("href")).toBe(
      "https://github.com/moritzbrantner/nlp-stack",
    );
  });
});
