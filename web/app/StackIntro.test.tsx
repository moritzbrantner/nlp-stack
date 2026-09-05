import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StackIntro } from "./StackIntro";

describe("StackIntro", () => {
  it("explains the Rust-to-Wasm browser boundary and links to the repository", () => {
    render(<StackIntro />);

    expect(screen.getByRole("heading", { level: 1, name: /Rust NLP package surfaces/i })).toBeTruthy();
    expect(screen.getByText(/Rust → wasm-bindgen → React → static Next\.js/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "View source" }).getAttribute("href")).toBe(
      "https://github.com/moritzbrantner/nlp-stack",
    );
  });
});
