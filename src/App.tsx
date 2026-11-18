
import React, { useMemo, useRef, useState } from "react";

const API_BASE_URL = (
  (import.meta as ImportMeta & {
    env?: { VITE_API_BASE_URL?: string };
  }).env?.VITE_API_BASE_URL?.trim() ||
  (window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : window.location.origin)
).replace(/\/+$/, "");

const MAX_BATCH_SIZE = 256;

const CLASSES = [
  {
    name: "Tree cover",
    color: "#2e8b45",
    image: "https://images.unsplash.com/photo-1448375240586-882707db888b?w=900&q=85",
    text: "Reveal the extent and continuity of tree cover across complex landscapes.",
  },
  {
    name: "Grassland",
    color: "#9ccc6c",
    image: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=900&q=85",
    text: "Understand open vegetation and natural grassland environments.",
  },
  {
    name: "Cropland",
    color: "#e6c430",
    image: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=900&q=85",
    text: "See agricultural patterns and the land that sustains us.",
  },
  {
    name: "Built-up",
    color: "#d04436",
    image: "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=900&q=85",
    text: "Distinguish developed environments from their natural surroundings.",
  },
  {
    name: "Permanent water",
    color: "#2a72c4",
    image: "https://images.unsplash.com/photo-1500375592092-40eb2168fd21?w=900&q=85",
    text: "Trace lakes, rivers and the water systems shaping our planet.",
  },
];

type Prediction = {
  class: string;
  confidence: number;
  probabilities: Record<string, number>;
  position?: {
    row: number;
    col: number;
    x?: number;
    y?: number;
    width?: number;
    height?: number;
  };
};

type ImageMetadata = {
  width?: number;
  height?: number;
  bands?: number;
  crs?: string;
  pixel_size_metres?: number;
  patch_size?: number;
  grid_rows?: number;
  grid_cols?: number;
  valid_patches?: number;
};

type ApiResult = {
  filename?: string;
  image_type?: string;
  preview_image?: string;
  segmentation_map?: string;
  preview_width?: number;
  preview_height?: number;
  model_name?: string;
  metadata?: ImageMetadata;
  total_patches?: number;
  total_pixels?: number;
  pixel_summary?: Record<string, number>;
  mean_probabilities?: Record<string, number>;
  patch_size?: number;
  patch_grid?: {
    rows: number;
    columns: number;
  };
  skipped_patches?: number;
  selected_bands?: Record<string, unknown> | null;
  predictions: Prediction[];
  summary: Record<string, number>;
  warning?: string;
};

function classColor(name: string) {
  return CLASSES.find((item) => item.name === name)?.color ?? "#c5cdc4";
}

function imageDimensions(result: ApiResult) {
  const width = result.metadata?.width ?? result.preview_width;
  const height = result.metadata?.height ?? result.preview_height;

  if (
    typeof width === "number" &&
    typeof height === "number" &&
    width > 0 &&
    height > 0
  ) {
    return `${width} × ${height} pixels`;
  }

  return null;
}

function probabilityFor(prediction: Prediction | undefined, name: string) {
  const raw = prediction?.probabilities?.[name];

  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    return null;
  }

  return raw;
}

function SyncedCompare({
  photo,
  mapSrc,
  grid,
  aspect,
  cover,
}: {
  photo: string;
  mapSrc?: string;
  grid?: {
    rows: number;
    columns: number;
    cells: Array<{ color: string | null; label: string }>;
  };
  aspect: number;
  cover?: { x: number; y: number };
}) {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const drag = useRef<{
    x: number;
    y: number;
    ox: number;
    oy: number;
  } | null>(null);

  function zoomBy(factor: number) {
    setScale((current) =>
      Math.min(8, Math.max(1, Math.round(current * factor * 100) / 100))
    );
  }

  function resetView() {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  }

  function pointerDown(event: React.PointerEvent<HTMLDivElement>) {
    drag.current = {
      x: event.clientX,
      y: event.clientY,
      ox: offset.x,
      oy: offset.y,
    };
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture is unavailable for this event. Drag still tracks movement.
    }
  }

  function pointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (!drag.current) return;
    const maxShift = 280 * scale;
    const nextX = drag.current.ox + (event.clientX - drag.current.x);
    const nextY = drag.current.oy + (event.clientY - drag.current.y);
    setOffset({
      x: Math.max(-maxShift, Math.min(maxShift, nextX)),
      y: Math.max(-maxShift, Math.min(maxShift, nextY)),
    });
  }

  function pointerUp() {
    drag.current = null;
  }

  const stageStyle = {
    transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
  };

  return (
    <div className="compare-board">
      <div className="compare-tools">
        <button type="button" onClick={() => zoomBy(1.25)}>
          Zoom in
        </button>
        <button type="button" onClick={() => zoomBy(0.8)}>
          Zoom out
        </button>
        <button type="button" onClick={resetView}>
          Reset
        </button>
        <span>{Math.round(scale * 100)}%</span>
      </div>

      <div className="compare-pair">
        <figure>
          <figcaption>Original satellite image</figcaption>
          <div
            className="compare-viewport"
            style={{ aspectRatio: aspect }}
            role="button"
            tabIndex={0}
            aria-label="Pan the original satellite image"
            onPointerDown={pointerDown}
            onPointerMove={pointerMove}
            onPointerUp={pointerUp}
            onPointerCancel={pointerUp}
          >
            <div className="compare-stage" style={stageStyle}>
              <img
                className="compare-photo"
                src={photo}
                alt="Original satellite image"
                draggable={false}
              />
            </div>
          </div>
        </figure>

        <figure>
          <figcaption>AI classification map</figcaption>
          <div
            className="compare-viewport"
            style={{ aspectRatio: aspect }}
            role="button"
            tabIndex={0}
            aria-label="Pan the classification map"
            onPointerDown={pointerDown}
            onPointerMove={pointerMove}
            onPointerUp={pointerUp}
            onPointerCancel={pointerUp}
          >
            <div className="compare-stage" style={stageStyle}>
              {mapSrc ? (
                <img
                  className="compare-map"
                  src={mapSrc}
                  alt="Classification map using the model's class colours"
                  draggable={false}
                />
              ) : grid ? (
                <div
                  className="compare-grid"
                  style={{
                    width: `${Math.min(1, cover?.x ?? 1) * 100}%`,
                    height: `${Math.min(1, cover?.y ?? 1) * 100}%`,
                    gridTemplateColumns: `repeat(${grid.columns}, minmax(0, 1fr))`,
                    gridTemplateRows: `repeat(${grid.rows}, minmax(0, 1fr))`,
                  }}
                  role="img"
                  aria-label="Classification grid of the returned patches"
                >
                  {grid.cells.map((cell, index) => (
                    <span
                      key={index}
                      title={cell.label}
                      style={{
                        background: cell.color ?? "#e7ebe3",
                      }}
                    />
                  ))}
                </div>
              ) : null}
            </div>
          </div>
          <div className="prediction-legend compare-legend">
            {CLASSES.map((item) => (
              <span key={item.name}>
                <i style={{ backgroundColor: item.color }} />
                {item.name}
              </span>
            ))}
          </div>
        </figure>
      </div>
    </div>
  );
}

function Brand() {
  return (
    <a className="brand" href="#top" aria-label="SatelliteVision AI home">
      <span className="brand-mark">
        <span />
      </span>
      <span>
        SatelliteVision <b>AI</b>
      </span>
    </a>
  );
}

function SectionLabel({
  children,
}: {
  children: React.ReactNode;
}) {
  return <div className="section-label">{children}</div>;
}

// =====================================================
// NUMPY FILE BATCHING
// =====================================================

async function createNpyBatches(
  file: File,
  maxBatchSize = MAX_BATCH_SIZE
): Promise<File[]> {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  const signature = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];

  if (!signature.every((value, index) => bytes[index] === value)) {
    throw new Error("Invalid NumPy file signature.");
  }

  const majorVersion = bytes[6];

  let headerStart: number;
  let headerLength: number;

  if (majorVersion === 1) {
    headerStart = 10;
    headerLength = new DataView(buffer).getUint16(8, true);
  } else if (majorVersion === 2 || majorVersion === 3) {
    headerStart = 12;
    headerLength = new DataView(buffer).getUint32(8, true);
  } else {
    throw new Error("Unsupported NPY file version.");
  }

  const dataOffset = headerStart + headerLength;

  if (dataOffset > bytes.length) {
    throw new Error("The NPY header is incomplete.");
  }

  const header = new TextDecoder().decode(
    bytes.slice(headerStart, dataOffset)
  );

  const shapeMatch = header.match(
    /(['"])shape\1\s*:\s*\(([^)]*)\)/
  );

  if (!shapeMatch) {
    throw new Error("Could not read the NPY file dimensions.");
  }

  const shape = shapeMatch[2]
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
    .map(Number);

  if (
    ![3, 4].includes(shape.length) ||
    !shape.every((value) => Number.isInteger(value) && value > 0)
  ) {
    throw new Error(
      "Expected NPY dimensions (3, 16, 16) or (N, 3, 16, 16)."
    );
  }

  const singlePatch = shape.length === 3;
  const totalPatches = singlePatch ? 1 : shape[0];

  const channels = singlePatch ? shape[0] : shape[1];
  const height = singlePatch ? shape[1] : shape[2];
  const width = singlePatch ? shape[2] : shape[3];

  if (channels !== 3 || height !== 16 || width !== 16) {
    throw new Error(
      `Unexpected NPY shape: (${shape.join(", ")}). ` +
      "Expected (3, 16, 16) or (N, 3, 16, 16)."
    );
  }

  if (/(['"])fortran_order\1\s*:\s*True/.test(header)) {
    throw new Error(
      "Fortran-order NPY files are not supported by this batcher."
    );
  }

  const dataLength = bytes.length - dataOffset;
  const bytesPerPatch = dataLength / totalPatches;

  if (
    totalPatches <= 0 ||
    dataLength <= 0 ||
    !Number.isInteger(bytesPerPatch)
  ) {
    throw new Error("The NPY file has an invalid data size.");
  }

  if (totalPatches <= maxBatchSize) {
    return [file];
  }

  const batches: File[] = [];

  for (let start = 0; start < totalPatches; start += maxBatchSize) {
    const end = Math.min(start + maxBatchSize, totalPatches);
    const batchCount = end - start;

    const newShape = [batchCount, ...shape.slice(1)];

    const updatedHeader = header.replace(
      shapeMatch[0],
      shapeMatch[0].replace(
        /\([^)]*\)/,
        `(${newShape.join(", ")})`
      )
    );

    const headerWithoutNewline = updatedHeader.trimEnd();

    const headerBaseBytes = new TextEncoder().encode(
      headerWithoutNewline
    );

    const padding =
      (16 -
        ((headerStart + headerBaseBytes.length + 1) % 16)) %
      16;

    const finalHeaderText =
      headerWithoutNewline + " ".repeat(padding) + "\n";

    const headerBytes = new TextEncoder().encode(finalHeaderText);

    const outputHeader = new Uint8Array(
      headerStart + headerBytes.length
    );

    outputHeader.set(bytes.slice(0, 8), 0);

    const view = new DataView(outputHeader.buffer);

    if (majorVersion === 1) {
      if (headerBytes.length > 65535) {
        throw new Error("The generated NPY header is too large.");
      }

      view.setUint16(8, headerBytes.length, true);
    } else {
      view.setUint32(8, headerBytes.length, true);
    }

    outputHeader.set(headerBytes, headerStart);

    const batchStart = dataOffset + start * bytesPerPatch;
    const batchEnd = dataOffset + end * bytesPerPatch;

    const batchData = bytes.slice(batchStart, batchEnd);

    const batchFile = new File(
      [outputHeader, batchData],
      `batch_${batches.length + 1}.npy`,
      { type: "application/octet-stream" }
    );

    batches.push(batchFile);
  }

  return batches;
}

// =====================================================
// SEND ONE FILE TO THE BACKEND
// =====================================================

async function sendPredictionRequest(
  file: File,
  endpoint: string
): Promise<ApiResult> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body,
  });

  let payload: any;

  try {
    payload = await response.json();
  } catch {
    throw new Error(
      "The backend returned an invalid response. Check that the API server is running."
    );
  }

  if (!response.ok) {
    throw new Error(
      payload.detail ||
      payload.message ||
      "The model could not process this file."
    );
  }

  const hasSegmentation =
    typeof payload.segmentation_map === "string" &&
    payload.segmentation_map.length > 0;
  const hasPredictions =
    Array.isArray(payload.predictions) &&
    payload.predictions.length > 0;

  if (!hasSegmentation && !hasPredictions) {
    throw new Error("The backend did not return any predictions.");
  }

  if (!Array.isArray(payload.predictions)) {
    payload.predictions = [];
  }

  return payload as ApiResult;
}

// =====================================================
// COMBINE BATCH RESULTS
// =====================================================

function combineBatchResults(
  results: ApiResult[],
  originalFilename: string
): ApiResult {
  const predictions: Prediction[] = [];
  const summary: Record<string, number> = {};

  let totalPatches = 0;
  let skippedPatches = 0;

  const warnings = new Set<string>();

  for (const item of results) {
    predictions.push(...item.predictions);

    totalPatches += Number(
      item.total_patches ?? item.predictions.length
    );

    skippedPatches += Number(item.skipped_patches ?? 0);

    for (const [className, count] of Object.entries(
      item.summary ?? {}
    )) {
      summary[className] =
        (summary[className] ?? 0) + Number(count ?? 0);
    }

    if (item.warning) {
      warnings.add(item.warning);
    }
  }

  return {
    filename: originalFilename,
    image_type: results[0]?.image_type ?? "NPY",
    preview_image: results[0]?.preview_image,
    model_name: results[0]?.model_name,
    total_patches: totalPatches,
    patch_size: results[0]?.patch_size ?? 16,
    skipped_patches: skippedPatches,
    selected_bands: results[0]?.selected_bands ?? null,
    predictions,
    summary,
    warning:
      warnings.size > 0
        ? Array.from(warnings).join(" ")
        : undefined,
  };
}

// =====================================================
// MAIN APP
// =====================================================

export default function App() {
  const fileInput = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ApiResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [selectedPatch, setSelectedPatch] = useState(0);

  const activePatchIndex =
    result && result.predictions.length > 0
      ? Math.min(selectedPatch, result.predictions.length - 1)
      : 0;

  const selectedPrediction = result?.predictions?.[activePatchIndex];

  const selectedProbabilities = useMemo(() => {
    return CLASSES.map((item) => ({
      ...item,
      value: probabilityFor(selectedPrediction, item.name),
    }));
  }, [selectedPrediction]);

  const probabilitiesAvailable = selectedProbabilities.some(
    (item) => item.value !== null
  );

  // =====================================================
  // CLASS DISTRIBUTION
  // =====================================================

  const classDistribution = useMemo(() => {
    if (!result) return [];

    const counts = result.segmentation_map
      ? result.pixel_summary ?? result.summary
      : result.summary;
    const total = result.segmentation_map
      ? Number(result.total_pixels ?? 0)
      : Number(result.total_patches ?? 0);

    if (!counts) return [];

    return CLASSES.map((item) => {
      const raw = counts[item.name];
      const count =
        typeof raw === "number" && Number.isFinite(raw) ? raw : null;

      return {
        ...item,
        count,
        percentage:
          count !== null && total > 0 ? (count / total) * 100 : null,
      };
    });
  }, [result]);

  const summaryReturned = classDistribution.some(
    (item) => item.count !== null
  );

  const highestCount = Math.max(
    0,
    ...classDistribution.map((item) => item.count ?? 0)
  );

  const majorityClasses = classDistribution.filter(
    (item) => item.count !== null && item.count === highestCount && highestCount > 0
  );

  const dimensions = result ? imageDimensions(result) : null;

  const hasSpatialGrid = Boolean(
    result?.patch_grid &&
      result.patch_grid.rows > 0 &&
      result.patch_grid.columns > 0 &&
      result.predictions.some(
        (item) =>
          item.position &&
          Number.isInteger(item.position.row) &&
          Number.isInteger(item.position.col)
      )
  );

  const patchesByCell = useMemo(() => {
    const cells = new Map<string, number>();

    result?.predictions.forEach((item, index) => {
      if (
        item.position &&
        Number.isInteger(item.position.row) &&
        Number.isInteger(item.position.col)
      ) {
        cells.set(`${item.position.row}-${item.position.col}`, index);
      }
    });

    return cells;
  }, [result]);

  // =====================================================
  // DOWNLOAD ALL WEBSITE PREDICTIONS
  // =====================================================

  function downloadSegmentationMap() {
    if (!result?.segmentation_map) return;

    const link = document.createElement("a");
    link.href = result.segmentation_map;
    link.download = "landcover_map.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function downloadWebsitePredictions() {
    if (!result?.predictions?.length && !result?.segmentation_map) {
      setError("There are no predictions available to download.");
      return;
    }

    const predictions = result.predictions.map((item) => ({
      class: item.class,
      confidence: item.confidence,
      probabilities: item.probabilities,
      position: item.position ?? null,
    }));

    const exportData = {
      filename: result.filename,
      image_type: result.image_type,
      model_name: result.model_name,
      metadata: result.metadata,
      total_patches: result.total_patches,
      total_pixels: result.total_pixels,
      patch_size: result.patch_size,
      patch_grid: result.patch_grid,
      pixel_summary: result.pixel_summary,
      mean_probabilities: result.mean_probabilities,
      predictions,
      summary: result.summary,
    };

    const blob = new Blob(
      [JSON.stringify(exportData, null, 2)],
      { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "website_predictions.json";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  }

  // =====================================================
  // FILE VALIDATION
  // =====================================================

  function chooseFile(next: File | undefined) {
    if (!next) return;

    setError("");
    setResult(null);
    setProgress(0);
    setSelectedPatch(0);

    const extension =
      next.name.toLowerCase().split(".").pop() || "";

    const supported = ["npy", "tif", "tiff"];

    if (!supported.includes(extension)) {
      setFile(null);

      setError(
        "Unsupported file. Please upload a raw .npy satellite patch " +
        "or a compatible .tif/.tiff satellite image. " +
        "Ordinary JPG and PNG photographs are not accepted."
      );

      return;
    }

    if (next.size > 50 * 1024 * 1024) {
      setFile(null);
      setError("The file is too large. Maximum size is 50 MB.");
      return;
    }

    setFile(next);
  }

  // =====================================================
  // DRAG AND DROP
  // =====================================================

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);

    chooseFile(event.dataTransfer.files?.[0]);
  }

  function handleInput(event: React.ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0]);
    event.currentTarget.value = "";
  }

  // =====================================================
  // MODEL CLASSIFICATION WITH AUTOMATIC BATCHING
  // =====================================================

  async function classify() {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);
    setProgress(0);
    setSelectedPatch(0);

    try {
      const extension =
        file.name.toLowerCase().split(".").pop() || "";

      if (extension === "npy") {
        const batches = await createNpyBatches(
          file,
          MAX_BATCH_SIZE
        );

        const batchResults: ApiResult[] = [];

        for (let index = 0; index < batches.length; index++) {
          const response = await sendPredictionRequest(
            batches[index],
            "/predict"
          );

          batchResults.push(response);

          setProgress(
            Math.round(((index + 1) / batches.length) * 100)
          );
        }

        const combined = combineBatchResults(
          batchResults,
          file.name
        );

        if (combined.predictions.length === 0) {
          throw new Error("No predictions were returned.");
        }

        setResult(combined);
      } else if (extension === "tif" || extension === "tiff") {
        const response = await sendPredictionRequest(
          file,
          "/predict-image"
        );

        setResult(response);
        setProgress(100);
      } else {
        throw new Error(
          "Only NPY and GeoTIFF satellite data are supported."
        );
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while contacting the model API."
      );
    } finally {
      setLoading(false);
    }
  }

  // =====================================================
  // WEBSITE
  // =====================================================

  return (
    <div className="site-shell" id="top">

      {/* HEADER */}

      <header className="site-header">
        <Brand />

        <nav className="desktop-nav" aria-label="Main navigation">
          <a href="#technology">Technology</a>
          <a href="#results">Results</a>
          <a href="#method">Method</a>
        </nav>

        <a className="nav-cta" href="#try-model">
          Try the model <span>↗</span>
        </a>
      </header>

      <main>

        {/* HERO */}

        <section className="hero section-wrap">
          <div className="hero-copy">
            <SectionLabel>
              AI-POWERED EARTH OBSERVATION
            </SectionLabel>

            <h1>
              See the Earth.
              <br />
              <em>Understand</em>
              <br />
              Every Pixel.
            </h1>

            <p>
              Turning satellite imagery into intelligent land-cover
              insights through advanced artificial intelligence.
            </p>

            <div className="hero-actions">
              <a className="button button-dark" href="#try-model">
                Explore AI Analysis <span>↗</span>
              </a>

              <a className="button button-light" href="#technology">
                Discover Our Technology
              </a>
            </div>

            <div className="hero-proof">
              <span>5 land-cover classes</span>
              <i />
              <span>Sentinel-2</span>
              <i />
              <span>86.50% test accuracy</span>
            </div>
          </div>

          <div className="hero-image-wrap">
            <img
              src="https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=1500&q=90"
              alt="Earth viewed from space"
            />

            <div className="image-shade" />

            <div className="image-coordinate coord-top">
              EARTH OBSERVATION <span>01 / 05</span>
            </div>

            <div className="image-coordinate coord-bottom">
              SENTINEL-2 · LAND-COVER INTELLIGENCE
            </div>

            <div className="scan-frame">
              <span />
              <span />
              <span />
              <span />
            </div>

            <div className="image-tag">
              <i /> REMOTE SENSING / AI
            </div>
          </div>
        </section>

        {/* INTRODUCTION */}

        <section className="intro-section section-wrap" id="technology">
          <div className="section-heading">
            <SectionLabel>01 / EARTH OBSERVATION</SectionLabel>
            <h2>
              A New Perspective
              <br />
              on Our Planet
            </h2>
          </div>

          <div className="intro-text">
            <p>
              Every landscape tells a story. We turn the detail of Earth
              observation into a clearer understanding of our changing world.
            </p>

            <span className="small-rule" />
          </div>
        </section>

        {/* LANDSCAPE CARDS */}

        <section className="landscape-grid section-wrap">
          <article className="landscape-card landscape-wide">
            <img src={CLASSES[0].image} alt="Forest canopy" />

            <div className="landscape-card-content">
              <span>01 — VEGETATION</span>
              <h3>Living forests</h3>
              <p>
                Reveal the extent and continuity of tree cover across
                complex landscapes.
              </p>
            </div>
          </article>

          <article className="landscape-card">
            <img src={CLASSES[2].image} alt="Agricultural fields" />

            <div className="landscape-card-content">
              <span>02 — AGRICULTURE</span>
              <h3>Productive landscapes</h3>
              <p>
                See agricultural patterns and the land that sustains us.
              </p>
            </div>
          </article>

          <article className="landscape-card">
            <img src={CLASSES[3].image} alt="City buildings" />

            <div className="landscape-card-content">
              <span>03 — DEVELOPMENT</span>
              <h3>Growing cities</h3>
              <p>
                Distinguish built-up environments from their natural
                surroundings.
              </p>
            </div>
          </article>

          <article className="landscape-card landscape-wide">
            <img src={CLASSES[4].image} alt="Water landscape" />

            <div className="landscape-card-content">
              <span>04 — WATER</span>
              <h3>Connected waters</h3>
              <p>
                Trace rivers, lakes and the water systems shaping our planet.
              </p>
            </div>
          </article>
        </section>

        {/* CLASSIFICATION */}

        <section className="classification-section">
          <div className="section-wrap classification-inner">
            <div className="classification-heading">
              <SectionLabel>02 / AI CLASSIFICATION</SectionLabel>

              <h2>
                From Satellite Pixels
                <br />
                to Intelligence
              </h2>

              <p>
                One landscape. Five meaningful classes. Translate visual
                complexity into structured land-cover information.
              </p>
            </div>

            <div className="class-visual">
              <div className="class-visual-image">
                <img
                  src="https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1400&q=85"
                  alt="Aerial agricultural landscape"
                />

                <div className="class-overlay">
                  <span className="map-cell cell-a" />
                  <span className="map-cell cell-b" />
                  <span className="map-cell cell-c" />
                  <span className="map-cell cell-d" />
                  <span className="map-cell cell-e" />
                </div>

                <div className="illustrative-note">
                  Illustrative visualization · Not a live model prediction
                </div>
              </div>

              <div className="class-legend">
                {CLASSES.map((item, index) => (
                  <div key={item.name}>
                    <i style={{ background: item.color }} />
                    <span>0{index + 1}</span>
                    <strong>{item.name}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* WORKFLOW */}

        <section className="workflow-section section-wrap" id="method">
          <div className="section-heading">
            <SectionLabel>03 / THE WORKFLOW</SectionLabel>

            <h2>
              A clear path from
              <br />
              image to insight
            </h2>
          </div>

          <div className="workflow-grid">
            <article>
              <span>01</span>
              <div className="workflow-icon">↗</div>
              <h3>Satellite Data</h3>
              <p>Upload or select satellite imagery.</p>
            </article>

            <article>
              <span>02</span>
              <div className="workflow-icon">⌘</div>
              <h3>AI Processing</h3>
              <p>
                A modified ResNet18 model analyzes satellite patches
                to distinguish five land-cover classes.
              </p>
            </article>

            <article>
              <span>03</span>
              <div className="workflow-icon">▤</div>
              <h3>Land-Cover Insights</h3>
              <p>
                Receive classification results and visual information
                that make the landscape easier to understand.
              </p>
            </article>
          </div>
        </section>

        {/* TECHNOLOGY */}

        <section className="technology-section">
          <div className="section-wrap technology-inner">
            <div className="technology-copy">
              <SectionLabel>04 / UNDER THE SURFACE</SectionLabel>

              <h2>
                Powered by
                <br />
                <em>Deep Learning</em>
              </h2>

              <p>
                Purpose-built for land-cover understanding. Grounded in
                satellite observation and a disciplined evaluation approach.
              </p>
            </div>

            <div className="technology-points">
              <article>
                <span>01</span>
                <div>
                  <h3>Architecture</h3>
                  <p>
                    Modified ResNet18 residual learning extracts visual
                    features from 16 × 16 satellite patches.
                  </p>
                </div>
              </article>

              <article>
                <span>02</span>
                <div>
                  <h3>Data</h3>
                  <p>
                    Sentinel-2 RGB satellite imagery with ESA WorldCover labels.
                  </p>
                </div>
              </article>

              <article>
                <span>03</span>
                <div>
                  <h3>Taxonomy</h3>
                  <p>
                    Five classes: tree cover, grassland, cropland,
                    built-up and permanent water.
                  </p>
                </div>
              </article>

              <article>
                <span>04</span>
                <div>
                  <h3>Validation</h3>
                  <p>
                    Training and evaluation are separated geographically
                    to assess generalization to held-out areas.
                  </p>
                </div>
              </article>
            </div>
          </div>
        </section>

        {/* RESULTS */}

        <section className="results-section section-wrap" id="results">
          <div className="results-intro">
            <SectionLabel>05 / RESEARCH &amp; EVALUATION</SectionLabel>

            <h2>
              Measured. Evaluated.
              <br />
              Transparent.
            </h2>

            <p>
              Results from the deployed selected-layer checkpoint.
              Performance is measured on 2,548 geographically held-out
              test patches and is not a guarantee for every landscape.
            </p>
          </div>

          <div className="metrics-grid">
            <article>
              <span>TEST ACCURACY</span>
              <strong>
                86.50<small>%</small>
              </strong>
              <p>
                Correct predictions across 2,548 held-out test patches.
              </p>
            </article>

            <article>
              <span>MACRO F1-SCORE</span>
              <strong>
                72.88<small>%</small>
              </strong>
              <p>
                Average F1-score across all five land-cover classes.
              </p>
            </article>

            <article>
              <span>MACRO RECALL</span>
              <strong>
                73.29<small>%</small>
              </strong>
              <p>
                Average recall across all five land-cover classes.
              </p>
            </article>
          </div>

          <div className="model-card">
            <div>
              <SectionLabel>MODEL CARD / SEED 42</SectionLabel>
              <h3>Evaluation at a glance</h3>
              <p>
                Modified ResNet18 · Selected-layer fine-tuning ·
                Best validation checkpoint at epoch 2
              </p>
            </div>

            <dl>
              <div>
                <dt>Architecture</dt>
                <dd>Modified ResNet18</dd>
              </div>

              <div>
                <dt>Fine-tuning</dt>
                <dd>Selected-layer</dd>
              </div>

              <div>
                <dt>Input</dt>
                <dd>16 × 16 RGB satellite patches</dd>
              </div>

              <div>
                <dt>Classes</dt>
                <dd>5 land-cover classes</dd>
              </div>

              <div>
                <dt>Labels</dt>
                <dd>ESA WorldCover</dd>
              </div>

              <div>
                <dt>Test set</dt>
                <dd>2,548 geographically held-out patches</dd>
              </div>

              <div>
                <dt>Test split</dt>
                <dd>Spatially separated geographic blocks</dd>
              </div>
            </dl>
          </div>

          <p className="results-footnote">
            Evaluation was conducted on 2,548 geographically held-out
            patches. Results should be interpreted in the context of this
            test sample and do not represent guaranteed performance in
            every geographic region.
          </p>

          <p className="results-footnote">
            Performance varies substantially by land-cover class.
            Grassland recognition remains a significant limitation,
            with a test recall of 10.53% (6 of 57 Grassland patches
            correctly classified). Overall accuracy should therefore
            be interpreted alongside macro-F1 and per-class recall.
          </p>
        </section>

        {/* INTERACTIVE MODEL DEMO */}

        <section className="try-section" id="try-model">
          <div
            className={`section-wrap try-inner${
              result ? " has-result" : ""
            }`}
          >
            <div className="try-heading">
              <SectionLabel>
                INTERACTIVE MODEL DEMONSTRATION
              </SectionLabel>

              <h2>
                Explore the Planet
                <br />
                <em>Through AI.</em>
              </h2>

              <p>
                Upload a raw Sentinel-2 NumPy patch or a compatible GeoTIFF
                satellite image to see the model's five-class predictions.
              </p>
            </div>

            <div className="try-panel">
              {result?.preview_image &&
              (result.segmentation_map || hasSpatialGrid) ? (
                <div className="compare-wrap">
                  <input
                    ref={fileInput}
                    type="file"
                    accept=".npy,.tif,.tiff"
                    hidden
                    onChange={handleInput}
                  />
                  <SyncedCompare
                    key={`${result.filename ?? "image"}-${result.total_pixels ?? result.total_patches ?? 0}`}
                    photo={result.preview_image}
                    mapSrc={result.segmentation_map}
                    aspect={
                      result.preview_width && result.preview_height
                        ? result.preview_width / result.preview_height
                        : result.patch_grid
                          ? result.patch_grid.columns / result.patch_grid.rows
                          : 1
                    }
                    cover={
                      !result.segmentation_map &&
                      result.patch_grid &&
                      result.patch_size &&
                      result.preview_width &&
                      result.preview_height
                        ? {
                            x:
                              (result.patch_grid.columns * result.patch_size) /
                              result.preview_width,
                            y:
                              (result.patch_grid.rows * result.patch_size) /
                              result.preview_height,
                          }
                        : undefined
                    }
                    grid={
                      result.segmentation_map || !result.patch_grid
                        ? undefined
                        : {
                            rows: result.patch_grid.rows,
                            columns: result.patch_grid.columns,
                            cells: Array.from(
                              {
                                length:
                                  result.patch_grid.rows *
                                  result.patch_grid.columns,
                              },
                              (_, index) => {
                                const row = Math.floor(
                                  index / result.patch_grid!.columns
                                );
                                const col = index % result.patch_grid!.columns;
                                const predictionIndex = patchesByCell.get(
                                  `${row}-${col}`
                                );
                                const prediction =
                                  predictionIndex === undefined
                                    ? undefined
                                    : result.predictions[predictionIndex];
                                return {
                                  color: prediction
                                    ? classColor(prediction.class)
                                    : null,
                                  label: prediction
                                    ? prediction.class
                                    : "Not classified",
                                };
                              }
                            ),
                          }
                    }
                  />
                  <div className="compare-actions">
                    <button
                      className="button button-light"
                      type="button"
                      onClick={() => fileInput.current?.click()}
                      disabled={loading}
                    >
                      Upload another image
                    </button>
                    <button
                      className="button button-dark"
                      type="button"
                      disabled={!file || loading}
                      onClick={classify}
                    >
                      {loading ? "Classifying…" : "Run AI classification"}
                    </button>
                    {result.segmentation_map && (
                      <button
                        className="button button-light"
                        type="button"
                        onClick={downloadSegmentationMap}
                      >
                        Download map
                      </button>
                    )}
                    <button
                      className="button button-light"
                      type="button"
                      onClick={downloadWebsitePredictions}
                    >
                      Download JSON
                    </button>
                  </div>
                  <p className="result-note">
                    {result.segmentation_map
                      ? "Drag either panel to pan. Zoom keeps both views on the same area. Class colours use the model’s own pixels and are not blended."
                      : "Drag either panel to pan. Each cell is one patch returned by the model, placed on the image area that patch covers. Empty cells were not classified."}
                  </p>
                </div>
              ) : (
              <>
              <div className="try-panel-top">
                <span>MODEL INFERENCE</span>
                <span className="api-status">
                  <i /> API CONNECTION
                </span>
              </div>

              {/* UPLOAD AREA */}

              <div
                className={`upload-area ${dragging ? "dragging" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInput}
                  type="file"
                  accept=".npy,.tif,.tiff"
                  hidden
                  onChange={handleInput}
                />

                <div className="upload-symbol">↑</div>

                <h3>
                  {file
                    ? file.name
                    : "Drop your satellite data here"}
                </h3>

                <p>
                  {file
                    ? "File selected and ready for validation."
                    : "or choose a file from your device"}
                </p>

                <button
                  className="button button-light"
                  type="button"
                  onClick={() => fileInput.current?.click()}
                  disabled={loading}
                >
                  Choose satellite data file
                </button>

                <small>
                  Accepted: NPY · GeoTIFF
                </small>
              </div>

              {/* INPUT GUIDANCE */}

              <div className="input-guidance">
                <span>SUPPORTED INPUTS</span>

                <strong>
                  Raw NPY satellite patches or compatible GeoTIFF
                </strong>

                <span>
                  NPY: (3, 16, 16) or (N, 3, 16, 16) ·
                  GeoTIFF: compatible RGB satellite bands ·
                  Up to 50 MB
                </span>
              </div>

              <p className="upload-disclaimer">
                JPG and PNG photographs are not accepted because display
                RGB values are not equivalent to calibrated Sentinel-2
                reflectance data.
              </p>

              {/* CLASSIFY BUTTON */}

              <button
                className="button button-dark classify-button"
                disabled={!file || loading}
                onClick={classify}
              >
                {loading
                  ? progress > 0
                    ? `Classifying… ${progress}%`
                    : "Classifying…"
                  : "Run AI classification"}
                <span>↗</span>
              </button>

              {loading && (
                <div className="classify-status" role="status">
                  <p>
                    {progress > 0
                      ? `${progress}% of this file has been classified.`
                      : "The model is reading this file. This request does not report a partial percentage."}
                  </p>
                  <div className="classify-track" aria-hidden="true">
                    <span
                      className={progress > 0 ? undefined : "is-wait"}
                      style={
                        progress > 0
                          ? { width: `${progress}%` }
                          : undefined
                      }
                    />
                  </div>
                </div>
              )}

              {/* ERRORS */}

              {error && (
                <p className="error-message" role="alert">
                  {error}
                </p>
              )}

              {/* RESULTS */}

              {result && (
                <div className="prediction-result" aria-live="polite">
                  {result.segmentation_map ? (
                    <div className="segmentation-view">
                      <div className="segmentation-pair">
                        <figure>
                          <figcaption>Uploaded image</figcaption>
                          {result.preview_image ? (
                            <img
                              src={result.preview_image}
                              alt="Uploaded satellite image"
                            />
                          ) : (
                            <p className="result-note">
                              No display image was returned.
                            </p>
                          )}
                        </figure>
                        <figure>
                          <figcaption>Land-cover prediction</figcaption>
                          <img
                            src={result.segmentation_map}
                            alt="Predicted land-cover class of each pixel"
                          />
                        </figure>
                      </div>

                      <p className="result-note">
                        Every valid pixel receives one class. The photograph is brightened so the scene is easier to see. Classification used the original file values.
                      </p>

                      <div className="prediction-legend">
                        {CLASSES.map((item) => (
                          <span key={item.name}>
                            <i style={{ backgroundColor: item.color }} />
                            {item.name}
                          </span>
                        ))}
                      </div>

                      <div className="prediction-headline">
                        <div>
                          <span>MOST COMMON CLASS</span>
                          <h3>
                            {majorityClasses.length
                              ? majorityClasses.map((item) => item.name).join(", ")
                              : "Not returned"}
                          </h3>
                          <p>Counted from the pixels in the map.</p>
                        </div>
                        <strong>
                          {majorityClasses[0]?.percentage != null ? (
                            <>
                              {majorityClasses[0].percentage.toFixed(1)}
                              <small>%</small>
                            </>
                          ) : (
                            <small>Not returned</small>
                          )}
                        </strong>
                      </div>

                      <h4>Average class probability</h4>
                      <p className="result-note">
                        Each bar is the average of that class score across the classified pixels. It is separate from the share of pixels painted in that color.
                      </p>
                      {result.mean_probabilities ? (
                        CLASSES.map((item) => {
                          const raw = result.mean_probabilities?.[item.name];
                          const value =
                            typeof raw === "number" && Number.isFinite(raw)
                              ? raw
                              : null;

                          return (
                            <div className="probability" key={item.name}>
                              <div>
                                <span>{item.name}</span>
                                <b>
                                  {value === null
                                    ? "Not returned"
                                    : `${value.toFixed(2)}%`}
                                </b>
                              </div>
                              <i>
                                <span
                                  style={{
                                    width:
                                      value === null
                                        ? "0%"
                                        : `${Math.max(0, Math.min(100, value))}%`,
                                    background: item.color,
                                  }}
                                />
                              </i>
                            </div>
                          );
                        })
                      ) : (
                        <p className="result-note">
                          Average class probabilities were not returned.
                        </p>
                      )}

                      <button
                        className="button button-light"
                        type="button"
                        onClick={() => fileInput.current?.click()}
                        disabled={loading}
                      >
                        Upload another image
                      </button>
                    </div>
                  ) : (
                  <div className="result-dashboard">
                    <section className="result-column">
                      <h4>Uploaded image</h4>

                      {result.preview_image ? (
                        <img
                          className="source-preview"
                          src={result.preview_image}
                          alt={
                            result.patch_grid
                              ? "Uploaded satellite image"
                              : "Display view of the first classified patch"
                          }
                        />
                      ) : (
                        <p className="result-note">
                          No display image was returned. The original file was still used for classification.
                        </p>
                      )}

                      <dl className="result-facts">
                        <div>
                          <dt>File</dt>
                          <dd>{result.filename || file?.name || "Not returned"}</dd>
                        </div>
                        <div>
                          <dt>Dimensions</dt>
                          <dd>{dimensions || "Not returned"}</dd>
                        </div>
                        <div>
                          <dt>Status</dt>
                          <dd>Classified</dd>
                        </div>
                      </dl>

                      <p className="result-note">
                        {result.patch_grid
                          ? "This is a display view of the uploaded image. Brightness is stretched so the scene is easier to see. The model used the original file values."
                          : result.preview_image
                            ? "This view shows the first patch only. The model did not return a picture of the whole file."
                            : "Upload another supported file to classify a different image."}
                      </p>

                      <button
                        className="button button-light"
                        type="button"
                        onClick={() => fileInput.current?.click()}
                        disabled={loading}
                      >
                        Upload another image
                      </button>
                    </section>

                    <section className="result-column">
                      <h4>Classification result</h4>

                      {hasSpatialGrid && result.patch_grid ? (
                        <div
                          className="prediction-grid"
                          style={{
                            gridTemplateColumns: `repeat(${result.patch_grid.columns}, minmax(0, 1fr))`,
                            aspectRatio: `${result.patch_grid.columns} / ${result.patch_grid.rows}`,
                          }}
                          role="grid"
                          aria-label="Classified 16 by 16 patches"
                        >
                          {Array.from(
                            { length: result.patch_grid.rows },
                            (_, row) =>
                              Array.from(
                                { length: result.patch_grid?.columns || 0 },
                                (_, col) => {
                                  const index = patchesByCell.get(`${row}-${col}`);
                                  const prediction =
                                    index === undefined
                                      ? undefined
                                      : result.predictions[index];

                                  if (!prediction || index === undefined) {
                                    return (
                                      <span
                                        key={`${row}-${col}`}
                                        className="patch-cell is-empty"
                                        title="This patch was not classified"
                                      />
                                    );
                                  }

                                  return (
                                    <button
                                      key={`${row}-${col}`}
                                      type="button"
                                      className={`patch-cell${
                                        index === activePatchIndex
                                          ? " is-selected"
                                          : ""
                                      }`}
                                      style={{
                                        backgroundColor: classColor(prediction.class),
                                      }}
                                      aria-pressed={index === activePatchIndex}
                                      aria-label={`Row ${row + 1}, column ${
                                        col + 1
                                      }, ${prediction.class}, ${
                                        typeof prediction.confidence === "number"
                                          ? `${prediction.confidence.toFixed(2)} percent confidence`
                                          : "confidence not returned"
                                      }`}
                                      title={`${prediction.class}${
                                        typeof prediction.confidence === "number"
                                          ? ` — ${prediction.confidence.toFixed(2)}% confidence`
                                          : ""
                                      }`}
                                      onClick={() => setSelectedPatch(index)}
                                    />
                                  );
                                }
                              )
                          )}
                        </div>
                      ) : (
                        <div
                          className="patch-strip"
                          role="list"
                          aria-label="Classified patches in the order returned by the model"
                        >
                          {result.predictions.map((prediction, index) => (
                            <button
                              key={`patch-${index}`}
                              type="button"
                              className={`patch-cell${
                                index === activePatchIndex ? " is-selected" : ""
                              }`}
                              style={{
                                backgroundColor: classColor(prediction.class),
                              }}
                              aria-pressed={index === activePatchIndex}
                              aria-label={`Patch ${index + 1}, ${prediction.class}`}
                              title={`${prediction.class}${
                                typeof prediction.confidence === "number"
                                  ? ` — ${prediction.confidence.toFixed(2)}% confidence`
                                  : ""
                              }`}
                              onClick={() => setSelectedPatch(index)}
                            />
                          ))}
                        </div>
                      )}

                      <p className="result-note">
                        {hasSpatialGrid
                          ? `Each square is one ${
                              result.patch_size ?? 16
                            }×${
                              result.patch_size ?? 16
                            } patch returned by the model. Empty squares were skipped. This is not a pixel-level map.`
                          : "Each square is one patch in the order returned by the model. These patches have no map position, so this is not a spatial classification."}
                      </p>

                      <div className="prediction-headline">
                        <div>
                          <span>
                            {result.predictions.length === 1
                              ? "PREDICTED CLASS"
                              : "MOST COMMON PATCH CLASS"}
                          </span>
                          <h3>
                            {result.predictions.length === 1
                              ? selectedPrediction?.class || "Not returned"
                              : majorityClasses.length
                                ? majorityClasses
                                    .map((item) => item.name)
                                    .join(", ")
                                : "Not returned"}
                          </h3>
                          <p>
                            {result.predictions.length === 1
                              ? "This file contains one classified patch."
                              : majorityClasses.length > 1
                                ? "These classes are tied for the largest number of patches."
                                : "This is the class assigned to the largest number of patches."}
                          </p>
                        </div>
                        <strong>
                          {result.predictions.length === 1 ? (
                            typeof selectedPrediction?.confidence === "number" ? (
                              <>
                                {selectedPrediction.confidence.toFixed(2)}
                                <small>%</small>
                              </>
                            ) : (
                              <small>Not returned</small>
                            )
                          ) : majorityClasses[0]?.count != null ? (
                            <>
                              {majorityClasses[0].count}
                              <small>/{result.total_patches}</small>
                            </>
                          ) : (
                            <small>Not returned</small>
                          )}
                        </strong>
                      </div>

                      {result.predictions.length > 1 && (
                        <p className="result-note">
                          Selected patch:{" "}
                          <b>{selectedPrediction?.class || "Not returned"}</b>
                          {selectedPrediction?.position
                            ? `, row ${selectedPrediction.position.row + 1}, column ${
                                selectedPrediction.position.col + 1
                              }`
                            : `, patch ${activePatchIndex + 1}`}
                          . Confidence:{" "}
                          <b>
                            {typeof selectedPrediction?.confidence === "number"
                              ? `${selectedPrediction.confidence.toFixed(2)}%`
                              : "not returned"}
                          </b>
                          . That percentage is this patch only. It is separate from the class probabilities below.
                        </p>
                      )}

                      <div className="prediction-legend">
                        {CLASSES.map((item) => (
                          <span key={item.name}>
                            <i style={{ backgroundColor: item.color }} />
                            {item.name}
                          </span>
                        ))}
                      </div>
                    </section>

                    <section className="result-column">
                      <h4>Class probabilities</h4>
                      <p className="result-note">
                        {selectedPrediction?.position
                          ? `Scores for the selected patch at row ${
                              selectedPrediction.position.row + 1
                            }, column ${selectedPrediction.position.col + 1}.`
                          : `Scores for patch ${activePatchIndex + 1}.`}
                        {" "}Select another square to see that patch.
                      </p>

                      {probabilitiesAvailable ? (
                        selectedProbabilities.map((item) => (
                          <div className="probability" key={item.name}>
                            <div>
                              <span>{item.name}</span>
                              <b>
                                {item.value === null
                                  ? "Not returned"
                                  : `${item.value.toFixed(2)}%`}
                              </b>
                            </div>
                            <i>
                              <span
                                style={{
                                  width:
                                    item.value === null
                                      ? "0%"
                                      : `${Math.max(0, Math.min(100, item.value))}%`,
                                  background: item.color,
                                }}
                              />
                            </i>
                          </div>
                        ))
                      ) : (
                        <p className="result-note">
                          Class probabilities were not returned for this patch.
                        </p>
                      )}
                    </section>
                  </div>
                  )}

                  <div className="result-cards">
                    <article>
                      <h4>Classification summary</h4>
                      <dl className="result-facts">
                        <div>
                          <dt>Predicted class</dt>
                          <dd>
                            {result.predictions.length === 1
                              ? selectedPrediction?.class || "Not returned"
                              : majorityClasses.length
                                ? majorityClasses.map((item) => item.name).join(", ")
                                : "Not returned"}
                          </dd>
                        </div>
                        <div>
                          <dt>
                            {result.segmentation_map ? "Share of pixels" : "Confidence"}
                          </dt>
                          <dd>
                            {result.segmentation_map
                              ? majorityClasses[0]?.percentage != null
                                ? majorityClasses.length > 1
                                  ? `Tied at ${majorityClasses[0].percentage.toFixed(1)}% of pixels each`
                                  : `${majorityClasses[0].percentage.toFixed(1)}% of classified pixels`
                                : "Not returned"
                              : typeof selectedPrediction?.confidence === "number"
                                ? `${selectedPrediction.confidence.toFixed(2)}% for the selected patch`
                                : "Not returned"}
                          </dd>
                        </div>
                        <div>
                          <dt>
                            {result.segmentation_map
                              ? "Classified pixels"
                              : "Classified patches"}
                          </dt>
                          <dd>
                            {result.segmentation_map
                              ? typeof result.total_pixels === "number"
                                ? result.total_pixels.toLocaleString()
                                : "Not returned"
                              : typeof result.total_patches === "number"
                                ? result.total_patches
                                : "Not returned"}
                            {typeof result.skipped_patches === "number"
                              ? ` · ${result.skipped_patches} skipped`
                              : ""}
                          </dd>
                        </div>
                      </dl>

                      {summaryReturned ? (
                        <div className="distribution-list">
                          {classDistribution.map((item) => (
                            <div className="distribution-item" key={item.name}>
                              <div className="distribution-heading">
                                <span>
                                  <i style={{ backgroundColor: item.color }} />
                                  {item.name}
                                </span>
                                <strong>
                                  {item.count === null
                                    ? "Not returned"
                                    : result.segmentation_map
                                      ? `${item.count.toLocaleString()} pixel${
                                          item.count === 1 ? "" : "s"
                                        }${
                                          item.percentage === null
                                            ? ""
                                            : ` · ${item.percentage.toFixed(1)}%`
                                        }`
                                      : `${item.count} patch${
                                        item.count === 1 ? "" : "es"
                                      }${
                                        item.percentage === null
                                          ? ""
                                          : ` · ${item.percentage.toFixed(1)}%`
                                      }`}
                                </strong>
                              </div>
                              <div className="distribution-bar">
                                <span
                                  style={{
                                    width: `${Math.max(
                                      0,
                                      Math.min(100, item.percentage ?? 0)
                                    )}%`,
                                    backgroundColor: item.color,
                                  }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="result-note">
                          A class distribution was not returned.
                        </p>
                      )}
                    </article>

                    <article>
                      <h4>Image information</h4>
                      <dl className="result-facts">
                        <div>
                          <dt>Filename</dt>
                          <dd>{result.filename || file?.name || "Not returned"}</dd>
                        </div>
                        <div>
                          <dt>Dimensions</dt>
                          <dd>{dimensions || "Not returned"}</dd>
                        </div>
                        <div>
                          <dt>Bands</dt>
                          <dd>
                            {typeof result.metadata?.bands === "number"
                              ? result.metadata.bands
                              : "Not returned"}
                          </dd>
                        </div>
                        <div>
                          <dt>Input format</dt>
                          <dd>{result.image_type || "Not returned"}</dd>
                        </div>
                        <div>
                          <dt>Model</dt>
                          <dd>{result.model_name || "Not returned by the model"}</dd>
                        </div>
                      </dl>
                    </article>

                    <article>
                      <h4>Download results</h4>
                      <p className="result-note">
                        {result.segmentation_map
                          ? "Save the pixel counts and average class probabilities returned for this image, or save the colored map."
                          : "Save the predictions returned for this file, including each patch class, its confidence, and its class probabilities when the model provided them."}
                      </p>
                      {result.segmentation_map && (
                        <button
                          className="button button-light"
                          type="button"
                          onClick={downloadSegmentationMap}
                        >
                          Download land-cover map
                        </button>
                      )}
                      <button
                        className="button button-light"
                        type="button"
                        onClick={downloadWebsitePredictions}
                      >
                        Download predictions JSON
                      </button>
                    </article>
                  </div>

                  {result.selected_bands && (
                    <p className="upload-disclaimer">
                      Selected bands:{" "}
                      {Object.entries(result.selected_bands)
                        .map(([key, value]) => `${key}: ${String(value)}`)
                        .join(" · ")}
                    </p>
                  )}

                  {result.warning && (
                    <p className="upload-disclaimer" role="note">
                      {result.warning}
                    </p>
                  )}

                  <p className="upload-disclaimer">
                    {result.segmentation_map
                      ? "The colored map is the model's class for each pixel. A large share of one color is not a confidence score. Compare the map with known labelled land cover."
                      : "Confidence is the model's score for the selected patch. It is not proof that the input is suitable or that the prediction is correct. Compare results with known labelled examples."}
                  </p>
                </div>
              )}

              <p className="upload-disclaimer">
                Scientific note: predictions depend on the input
                reflectance scale, spatial resolution, channel order
                and preprocessing matching the model's training data.
                The backend applies the model's stored normalization.
                Do not manually rescale reflectance values unless
                the model's input requirements specify it.
              </p>
              </>
              )}
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}

      <footer className="site-footer section-wrap">
        <div>
          <Brand />
          <p>
            AI-powered understanding of our changing planet.
          </p>
        </div>

        <div className="footer-links">
          <a href="#technology">Technology</a>
          <a href="#results">Results</a>
          <a href="#method">Method</a>
          <a href="#try-model">Try the model</a>
        </div>

        <div className="footer-note">
          Contains modified Copernicus Sentinel data.
          <br />
          Land-cover labels © ESA WorldCover
          <br />
          Research by Yi Nuo
          <br />
          © 2026 SatelliteVision AI · Research, not a live geospatial service.
        </div>
      </footer>
    </div>
  );
}
