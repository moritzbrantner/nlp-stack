export type OcrLanguage = "eng" | "deu" | "eng+deu" | "spa";

export interface BrowserTextIngestOptions {
  ocrLanguage: OcrLanguage;
  ocrScannedPdfPages?: boolean;
}

export interface BrowserTextIngestResult {
  text: string;
  sourceLabel: string;
  method: "text" | "markup" | "pdf" | "pdf+ocr" | "ocr";
  pageCount?: number;
  ocrPageCount?: number;
}

type ProgressReporter = (message: string) => void;

type OcrWorker = {
  recognize: (image: File | HTMLCanvasElement) => Promise<{ data: { text: string } }>;
  terminate: () => Promise<void>;
};

type TesseractApi = {
  createWorker: (
    languages: string | string[],
    oem?: number,
    options?: { logger?: (message: { status?: string; progress?: number }) => void },
  ) => Promise<OcrWorker>;
};

type PdfTextItem = { str?: string; hasEOL?: boolean };
type PdfPage = {
  getTextContent: () => Promise<{ items: PdfTextItem[] }>;
  getViewport: (options: { scale: number }) => { width: number; height: number };
  render: (options: {
    canvasContext: CanvasRenderingContext2D;
    viewport: { width: number; height: number };
  }) => { promise: Promise<void> };
};
type PdfDocument = {
  numPages: number;
  getPage: (pageNumber: number) => Promise<PdfPage>;
};
type PdfJsApi = {
  GlobalWorkerOptions: { workerSrc: string };
  getDocument: (options: { data: Uint8Array }) => { promise: Promise<PdfDocument> };
};

declare global {
  interface Window {
    Tesseract?: TesseractApi;
    pdfjsLib?: PdfJsApi;
  }
}

const TESSERACT_SCRIPT = "https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js";
const PDF_JS_SCRIPT = "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js";
const PDF_JS_WORKER = "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";
const MAX_FILE_BYTES = 64 * 1024 * 1024;
const MAX_PDF_PAGES = 100;
const TEXT_EXTENSIONS = new Set([
  "txt",
  "md",
  "markdown",
  "csv",
  "tsv",
  "json",
  "yaml",
  "yml",
  "log",
  "srt",
  "vtt",
]);
const MARKUP_EXTENSIONS = new Set(["html", "htm", "xml", "xhtml"]);
const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"]);
const scriptLoads = new Map<string, Promise<void>>();

export const browserTextFileAccept = [
  ".txt",
  ".md",
  ".markdown",
  ".csv",
  ".tsv",
  ".json",
  ".yaml",
  ".yml",
  ".log",
  ".srt",
  ".vtt",
  ".html",
  ".htm",
  ".xml",
  ".xhtml",
  ".pdf",
  "image/*",
].join(",");

export function normalizeExtractedText(text: string): string {
  return text
    .replace(/\r\n?/g, "\n")
    .replace(/[\t ]+\n/g, "\n")
    .replace(/[\t ]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function extractMarkupText(markup: string, mimeType = "text/html"): string {
  const documentNode = new DOMParser().parseFromString(markup, mimeType);
  if (mimeType === "text/html") {
    for (const node of documentNode.querySelectorAll("script, style, noscript, template")) {
      node.remove();
    }
  }
  return normalizeExtractedText(documentNode.body?.textContent ?? documentNode.documentElement.textContent ?? "");
}

export async function ingestBrowserFile(
  file: File,
  options: BrowserTextIngestOptions,
  reportProgress: ProgressReporter = () => undefined,
): Promise<BrowserTextIngestResult> {
  if (file.size > MAX_FILE_BYTES) {
    throw new Error("Files larger than 64 MiB are not processed in the browser workbench.");
  }

  const extension = fileExtension(file.name);
  if (file.type === "application/pdf" || extension === "pdf") {
    return extractPdf(file, options, reportProgress);
  }

  if (file.type.startsWith("image/") || IMAGE_EXTENSIONS.has(extension)) {
    reportProgress("Loading the local OCR runtime…");
    const worker = await createOcrWorker(options.ocrLanguage, reportProgress);
    try {
      reportProgress(`Recognizing text in ${file.name}…`);
      const result = await worker.recognize(file);
      const text = normalizeExtractedText(result.data.text);
      if (!text) {
        throw new Error("OCR completed but did not find readable text.");
      }
      return { text, sourceLabel: file.name, method: "ocr", ocrPageCount: 1 };
    } finally {
      await worker.terminate();
    }
  }

  if (MARKUP_EXTENSIONS.has(extension) || file.type === "text/html" || file.type.includes("xml")) {
    reportProgress(`Parsing ${file.name}…`);
    const source = await file.text();
    const mimeType = extension === "xml" || extension === "xhtml" || file.type.includes("xml")
      ? "application/xml"
      : "text/html";
    const text = extractMarkupText(source, mimeType);
    if (!text) {
      throw new Error("The uploaded markup did not contain readable text.");
    }
    return { text, sourceLabel: file.name, method: "markup" };
  }

  if (TEXT_EXTENSIONS.has(extension) || file.type.startsWith("text/") || file.type === "application/json") {
    reportProgress(`Reading ${file.name}…`);
    const text = normalizeExtractedText(await file.text());
    if (!text) {
      throw new Error("The uploaded file is empty.");
    }
    return { text, sourceLabel: file.name, method: "text" };
  }

  throw new Error(
    "Unsupported file type. Upload text/Markdown/CSV/JSON/XML/HTML, a PDF, or an image for OCR.",
  );
}

async function extractPdf(
  file: File,
  options: BrowserTextIngestOptions,
  reportProgress: ProgressReporter,
): Promise<BrowserTextIngestResult> {
  reportProgress("Loading the PDF parser…");
  const pdfjs = await loadPdfJs();
  const data = new Uint8Array(await file.arrayBuffer());
  const documentNode = await pdfjs.getDocument({ data }).promise;
  if (documentNode.numPages > MAX_PDF_PAGES) {
    throw new Error(`This PDF has ${documentNode.numPages} pages; the browser workbench currently processes up to ${MAX_PDF_PAGES}.`);
  }

  const pages: string[] = [];
  let worker: OcrWorker | null = null;
  let ocrPageCount = 0;
  try {
    for (let pageNumber = 1; pageNumber <= documentNode.numPages; pageNumber += 1) {
      reportProgress(`Extracting PDF text — page ${pageNumber} of ${documentNode.numPages}…`);
      const page = await documentNode.getPage(pageNumber);
      const textContent = await page.getTextContent();
      let pageText = normalizeExtractedText(
        textContent.items
          .map((item) => `${item.str ?? ""}${item.hasEOL ? "\n" : " "}`)
          .join(""),
      );

      if ((options.ocrScannedPdfPages ?? true) && pageText.length < 20) {
        worker ??= await createOcrWorker(options.ocrLanguage, reportProgress);
        reportProgress(`No text layer on page ${pageNumber}; running OCR…`);
        pageText = await ocrPdfPage(page, worker);
        if (pageText) {
          ocrPageCount += 1;
        }
      }

      if (pageText) {
        pages.push(pageText);
      }
    }
  } finally {
    if (worker) {
      await worker.terminate();
    }
  }

  const text = normalizeExtractedText(pages.join("\n\n"));
  if (!text) {
    throw new Error("No readable text could be extracted from the PDF.");
  }

  return {
    text,
    sourceLabel: file.name,
    method: ocrPageCount > 0 ? "pdf+ocr" : "pdf",
    pageCount: documentNode.numPages,
    ocrPageCount,
  };
}

async function ocrPdfPage(page: PdfPage, worker: OcrWorker): Promise<string> {
  const viewport = page.getViewport({ scale: 1.75 });
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.ceil(viewport.width));
  canvas.height = Math.max(1, Math.ceil(viewport.height));
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("The browser could not create a canvas for PDF OCR.");
  }
  await page.render({ canvasContext: context, viewport }).promise;
  const result = await worker.recognize(canvas);
  return normalizeExtractedText(result.data.text);
}

async function createOcrWorker(
  language: OcrLanguage,
  reportProgress: ProgressReporter,
): Promise<OcrWorker> {
  await loadScript("nlp-stack-tesseract", TESSERACT_SCRIPT);
  const tesseract = window.Tesseract;
  if (!tesseract) {
    throw new Error("The OCR runtime loaded without exposing Tesseract.");
  }
  const languages = language.split("+");
  return tesseract.createWorker(languages, undefined, {
    logger: (message) => {
      if (!message.status) {
        return;
      }
      const percent = typeof message.progress === "number" ? ` ${Math.round(message.progress * 100)}%` : "";
      reportProgress(`OCR: ${message.status}${percent}`);
    },
  });
}

async function loadPdfJs(): Promise<PdfJsApi> {
  await loadScript("nlp-stack-pdfjs", PDF_JS_SCRIPT);
  const pdfjs = window.pdfjsLib;
  if (!pdfjs) {
    throw new Error("The PDF parser loaded without exposing pdfjsLib.");
  }
  pdfjs.GlobalWorkerOptions.workerSrc = PDF_JS_WORKER;
  return pdfjs;
}

function loadScript(id: string, src: string): Promise<void> {
  const cached = scriptLoads.get(id);
  if (cached) {
    return cached;
  }

  const existing = document.getElementById(id) as HTMLScriptElement | null;
  if (existing?.dataset.ready === "true") {
    return Promise.resolve();
  }

  const promise = new Promise<void>((resolve, reject) => {
    const script = existing ?? document.createElement("script");
    const onLoad = () => {
      script.dataset.ready = "true";
      resolve();
    };
    const onError = () => reject(new Error(`Unable to load browser dependency: ${src}`));
    script.addEventListener("load", onLoad, { once: true });
    script.addEventListener("error", onError, { once: true });
    if (!existing) {
      script.id = id;
      script.src = src;
      script.async = true;
      document.head.append(script);
    }
  });
  scriptLoads.set(id, promise);
  return promise;
}

function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot + 1).toLowerCase() : "";
}
