
import io
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import rasterio

from PIL import Image
from rasterio.io import MemoryFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision.models import resnet18
from unet import load_unet, segment_image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "satellitevision_final.pth"
UNET_PATH = BASE_DIR / "models" / "landcover_unet.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 16
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PATCHES = 256
MAX_PIXELS = 100_000_000

DEFAULT_CLASSES = [
    "Tree cover",
    "Grassland",
    "Cropland",
    "Built-up",
    "Permanent water",
]


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SatelliteVision AI",
    description="Five-class satellite land-cover classification API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SatelliteResNet18(nn.Module):

    def __init__(self, num_classes=5, mean=None, std=None):
        super().__init__()

        base = resnet18(weights=None)

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = nn.Identity()

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

        self.register_buffer(
            "mean",
            torch.tensor(
                mean or [
                    0.08995474,
                    0.09674594,
                    0.08033189
                ],
                dtype=torch.float32
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                std or [
                    0.04955327,
                    0.03809398,
                    0.03873659
                ],
                dtype=torch.float32
            ).view(1, 3, 1, 1)
        )

    def forward(self, x):

        x = (x - self.mean) / self.std

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    required = [
        "model_state_dict",
        "architecture",
        "class_names",
        "normalization_mean",
        "normalization_std",
    ]

    for key in required:
        if key not in checkpoint:
            raise RuntimeError(
                f"Checkpoint missing key: {key}"
            )

    if checkpoint["architecture"] != (
        "resnet18_small_image_selected_layer"
    ):
        raise RuntimeError(
            "Unexpected architecture: "
            + str(checkpoint["architecture"])
        )

    classes = checkpoint["class_names"]

    if classes != DEFAULT_CLASSES:
        raise RuntimeError(
            f"Unexpected class order: {classes}"
        )

    mean = checkpoint["normalization_mean"]
    std = checkpoint["normalization_std"]

    model = SatelliteResNet18(
        num_classes=len(classes),
        mean=mean,
        std=std
    )

    state_dict = checkpoint["model_state_dict"].copy()

    state_dict["mean"] = torch.tensor(
        mean,
        dtype=torch.float32
    ).view(1, 3, 1, 1)

    state_dict["std"] = torch.tensor(
        std,
        dtype=torch.float32
    ).view(1, 3, 1, 1)

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    metadata = {
        "architecture": checkpoint["architecture"],
        "model_type": checkpoint.get("model_type"),
        "class_names": classes,
        "input_size": checkpoint.get("input_size", PATCH_SIZE),
        "input_channels": checkpoint.get("input_channels", 3),
        "normalization_mean": mean,
        "normalization_std": std,
        "reflectance_scale": checkpoint.get("reflectance_scale"),
        "patch_size_metres": checkpoint.get("patch_size_metres"),
        "minimum_patch_purity": checkpoint.get("minimum_patch_purity"),
        "seed": checkpoint.get("seed"),
        "strategy": checkpoint.get("strategy"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_macro_f1": checkpoint.get(
            "best_validation_macro_f1"
        ),
        "test_accuracy": checkpoint.get("test_accuracy"),
        "test_macro_f1": checkpoint.get("test_macro_f1"),
        "test_macro_recall": checkpoint.get("test_macro_recall"),
    }

    print("======================================")
    print("SatelliteVision AI model loaded")
    print("Architecture:", metadata["architecture"])
    print("Classes:", classes)
    print("Device:", DEVICE)
    print("Checkpoint:", MODEL_PATH)
    print("======================================")

    return model, metadata


model, MODEL_METADATA = load_model()
CLASS_NAMES = MODEL_METADATA["class_names"]

if UNET_PATH.exists():
    segmentation_model, SEGMENTATION_METADATA = load_unet(UNET_PATH, DEVICE)
else:
    segmentation_model = None
    SEGMENTATION_METADATA = None


# ============================================================
# IMAGE PREVIEW
# ============================================================

CLASS_RGB = {
    "Tree cover": (46, 139, 69),
    "Grassland": (156, 204, 108),
    "Cropland": (230, 196, 48),
    "Built-up": (208, 68, 54),
    "Permanent water": (42, 114, 196),
}


def stretch_to_uint8(image_array):

    image = np.moveaxis(
        image_array,
        0,
        -1
    ).astype(np.float32)

    output = np.zeros_like(image)

    for channel in range(3):

        band = image[:, :, channel]

        if not np.isfinite(band).any():
            continue

        low = np.nanpercentile(band, 2)
        high = np.nanpercentile(band, 98)

        if high > low:
            output[:, :, channel] = (
                np.clip(
                    (band - low) / (high - low),
                    0,
                    1
                ) * 255
            )

    return output.astype(np.uint8)


def upscale_for_display(image_uint8, min_long_side=1600, max_long_side=2400):

    height, width = image_uint8.shape[:2]
    long_side = max(height, width)

    if long_side == 0 or long_side >= min_long_side:
        return image_uint8, 1

    scale = int(np.ceil(min_long_side / long_side))
    scale = max(1, min(scale, max_long_side // long_side))

    enlarged = Image.fromarray(image_uint8).resize(
        (width * scale, height * scale),
        Image.Resampling.NEAREST,
    )

    return np.asarray(enlarged), scale


def encode_png(image_uint8):

    buffer = io.BytesIO()

    Image.fromarray(image_uint8).save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return "data:image/png;base64," + encoded


def colorize_prediction(prediction):

    canvas = np.full(
        (prediction.shape[0], prediction.shape[1], 3),
        231,
        dtype=np.uint8,
    )

    for index, name in enumerate(CLASS_NAMES):
        canvas[prediction == index] = CLASS_RGB[name]

    return canvas


def create_preview(image_array):

    display, _scale = upscale_for_display(
        stretch_to_uint8(image_array)
    )

    return encode_png(display)


def class_color(prediction):

    name = prediction.get("class")
    rgb = CLASS_RGB.get(name)

    if rgb is None:
        return (180, 180, 180)

    return rgb


def choose_map_stride(height, width, max_windows=1800):

    if height < PATCH_SIZE or width < PATCH_SIZE:
        return PATCH_SIZE

    span_y = height - PATCH_SIZE + 1
    span_x = width - PATCH_SIZE + 1
    stride = 1

    while stride < PATCH_SIZE:
        rows = (span_y + stride - 1) // stride
        cols = (span_x + stride - 1) // stride
        if rows * cols <= max_windows:
            return stride
        stride += 1

    return PATCH_SIZE


def fill_uncovered(canvas, covered):

    for _ in range(PATCH_SIZE):

        if covered.all():
            break

        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):

            shifted_cover = np.zeros_like(covered)
            shifted_color = np.zeros_like(canvas)

            if axis == 0 and sign == 1:
                shifted_cover[1:] = covered[:-1]
                shifted_color[1:] = canvas[:-1]
            elif axis == 0 and sign == -1:
                shifted_cover[:-1] = covered[1:]
                shifted_color[:-1] = canvas[1:]
            elif axis == 1 and sign == 1:
                shifted_cover[:, 1:] = covered[:, :-1]
                shifted_color[:, 1:] = canvas[:, :-1]
            else:
                shifted_cover[:, :-1] = covered[:, 1:]
                shifted_color[:, :-1] = canvas[:, 1:]

            take = ~covered & shifted_cover
            canvas[take] = shifted_color[take]
            covered |= take

    return canvas


def paint_solid_map(height, width, origins, predictions, stride):

    canvas = np.full((height, width, 3), 232, dtype=np.uint8)
    covered = np.zeros((height, width), dtype=bool)
    half = PATCH_SIZE // 2

    for (y, x), prediction in zip(origins, predictions):

        color = np.array(class_color(prediction), dtype=np.uint8)
        center_y = y + half
        center_x = x + half
        y1 = max(0, center_y - stride // 2)
        x1 = max(0, center_x - stride // 2)
        y2 = min(height, y1 + stride)
        x2 = min(width, x1 + stride)
        canvas[y1:y2, x1:x2] = color
        covered[y1:y2, x1:x2] = True

    if covered.any():
        fill_uncovered(canvas, covered)

    return canvas


def dense_map_inputs(rgb, valid):

    height, width = valid.shape
    stride = choose_map_stride(height, width)
    windows = []
    origins = []

    for y in range(0, height - PATCH_SIZE + 1, stride):

        for x in range(0, width - PATCH_SIZE + 1, stride):

            window = rgb[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            window_valid = valid[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            if not window_valid.all():
                continue

            if not np.isfinite(window).all():
                continue

            if window.min() < 0 or window.max() > 2:
                continue

            windows.append(window)
            origins.append((int(y), int(x)))

    return stride, windows, origins


# ============================================================
# PREDICTION
# ============================================================

def predict_patches(patches):

    results = []
    batch_size = 32

    with torch.inference_mode():

        for start in range(0, len(patches), batch_size):

            batch = patches[
                start:start + batch_size
            ]

            tensor = torch.from_numpy(
                np.asarray(batch, dtype=np.float32)
            ).to(DEVICE)

            logits = model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predicted = probabilities.argmax(dim=1)

            for i in range(len(batch)):

                class_index = int(predicted[i].item())

                confidence = float(
                    probabilities[i, class_index].item()
                )

                results.append({
                    "class_index": class_index,

                    "class_name": CLASS_NAMES[class_index],
                    "class": CLASS_NAMES[class_index],

                    "confidence": round(
                        confidence * 100,
                        2
                    ),

                    "probabilities": {
                        CLASS_NAMES[j]: round(
                            float(probabilities[i, j].item()) * 100,
                            2
                        )
                        for j in range(len(CLASS_NAMES))
                    }
                })

    return results


# ============================================================
# SUMMARY
# ============================================================

def create_summary(predictions):

    counts = {
        name: 0 for name in CLASS_NAMES
    }

    for prediction in predictions:
        counts[prediction["class_name"]] += 1

    return counts


# ============================================================
# HEALTH AND MODEL INFO
# ============================================================

@app.get("/")
def root():

    return {
        "message": "SatelliteVision AI API",
        "status": "running",
        "device": str(DEVICE)
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "segmentation_loaded": segmentation_model is not None,
        "device": str(DEVICE)
    }


@app.get("/model-info")
def model_info():

    return {
        "model_name": "SatelliteVision AI",
        "metadata": MODEL_METADATA,
        "classes": CLASS_NAMES,
        "device": str(DEVICE)
    }


# ============================================================
# NUMPY PATCH PREDICTION
# ============================================================

@app.post("/predict")
async def predict_npy(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Upload a .npy file."
        )

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 50 MB."
        )

    try:

        array = np.load(
            io.BytesIO(contents),
            allow_pickle=False
        )

        if array.ndim == 3:
            array = array[np.newaxis, ...]

        if (
            array.ndim != 4
            or array.shape[1:] != (3, PATCH_SIZE, PATCH_SIZE)
        ):
            raise ValueError(
                "Expected shape (N, 3, 16, 16)."
            )

        if len(array) == 0 or len(array) > MAX_PATCHES:
            raise ValueError(
                f"Upload between 1 and {MAX_PATCHES} patches."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "Input contains invalid values."
            )

        array = array.astype(np.float32)

        if array.min() < 0 or array.max() > 2:
            raise ValueError(
                "Input values must be between 0 and 2."
            )

        predictions = predict_patches(array)

        # CREATE PREVIEW FROM THE FIRST UPLOADED PATCH
        preview_image = create_preview(array[0])

        return {
            "success": True,
            "filename": file.filename,
            "image_type": "NumPy satellite patches",
            "total_patches": len(predictions),
            "patch_size": PATCH_SIZE,
            "skipped_patches": 0,
            "predictions": predictions,
            "summary": create_summary(predictions),
            "preview_image": preview_image,
            "model_name": MODEL_METADATA["architecture"]
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# GEOTIFF PROCESSING
# ============================================================

def process_geotiff(contents, collect_patches=True):

    with MemoryFile(contents) as memory_file:

        with memory_file.open() as dataset:

            if dataset.count < 3:
                raise ValueError(
                    "GeoTIFF needs at least 3 bands."
                )

            height = dataset.height
            width = dataset.width

            if height * width > MAX_PIXELS:
                raise ValueError(
                    "Image exceeds maximum pixel limit."
                )

            image = dataset.read(
                out_dtype="float32"
            )

            descriptions = [
                (description or "").lower()
                for description in dataset.descriptions
            ]

            rgb_indices = None

            if descriptions:

                red = [
                    i for i, name in enumerate(descriptions)
                    if "red" in name or name == "b04"
                ]

                green = [
                    i for i, name in enumerate(descriptions)
                    if "green" in name or name == "b03"
                ]

                blue = [
                    i for i, name in enumerate(descriptions)
                    if "blue" in name or name == "b02"
                ]

                if red and green and blue:
                    rgb_indices = [
                        red[0],
                        green[0],
                        blue[0]
                    ]

            if rgb_indices is None:

                if dataset.count >= 4:
                    # Assumes B02, B03, B04, B08 ordering.
                    rgb_indices = [2, 1, 0]
                else:
                    rgb_indices = [0, 1, 2]

            rgb = image[rgb_indices]

            valid = np.isfinite(rgb).all(axis=0)

            if not valid.any():
                raise ValueError(
                    "No valid pixels found."
                )

            scale = MODEL_METADATA["reflectance_scale"]

            if scale and np.nanpercentile(rgb, 99) > 2:
                rgb = rgb / float(scale)

            rows = height // PATCH_SIZE
            cols = width // PATCH_SIZE

            if collect_patches and rows * cols > MAX_PATCHES:
                raise ValueError(
                    f"Image generates {rows * cols} patches. "
                    f"Maximum is {MAX_PATCHES}. Upload a smaller area."
                )

            if not collect_patches and height * width > 1_500_000:
                raise ValueError(
                    "This image is too large for the pixel model. "
                    "Upload an area up to about 1200 by 1200 pixels."
                )

            patches = []
            positions = []

            if not collect_patches:
                pixel_size = abs(float(dataset.res[0])) if dataset.res else None
                return {
                    "patches": np.empty((0, 3, PATCH_SIZE, PATCH_SIZE), np.float32),
                    "positions": [],
                    "rgb": rgb,
                    "valid": valid,
                    "metadata": {
                        "width": width,
                        "height": height,
                        "bands": dataset.count,
                        "crs": str(dataset.crs),
                        "pixel_size_metres": pixel_size,
                        "patch_size": PATCH_SIZE,
                        "grid_rows": rows,
                        "grid_cols": cols,
                        "valid_patches": 0,
                    },
                }

            for row in range(rows):

                for col in range(cols):

                    y1 = row * PATCH_SIZE
                    y2 = y1 + PATCH_SIZE

                    x1 = col * PATCH_SIZE
                    x2 = x1 + PATCH_SIZE

                    patch = rgb[:, y1:y2, x1:x2]
                    patch_valid = valid[y1:y2, x1:x2]

                    if not patch_valid.all():
                        continue

                    if not np.isfinite(patch).all():
                        continue

                    if patch.min() < 0 or patch.max() > 2:
                        continue

                    patches.append(patch)

                    positions.append({
                        "row": row,
                        "col": col,
                        "x": x1,
                        "y": y1,
                        "width": PATCH_SIZE,
                        "height": PATCH_SIZE
                    })

            if not patches:
                raise ValueError(
                    "No valid complete 16x16 patches found."
                )

            return {
                "patches": np.asarray(
                    patches,
                    dtype=np.float32
                ),
                "positions": positions,
                "rgb": rgb,
                "valid": valid,
                "metadata": {
                    "width": width,
                    "height": height,
                    "bands": dataset.count,
                    "crs": str(dataset.crs),
                    "pixel_size_metres": abs(float(dataset.res[0])) if dataset.res else None,
                    "patch_size": PATCH_SIZE,
                    "grid_rows": rows,
                    "grid_cols": cols,
                    "valid_patches": len(patches)
                }
            }


# ============================================================
# GEOTIFF PREDICTION
# ============================================================

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=400,
            detail="Upload a GeoTIFF (.tif or .tiff)."
        )

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 50 MB."
        )

    try:

        processed = process_geotiff(
            contents,
            collect_patches=segmentation_model is None,
        )

        if segmentation_model is not None:
            class_map, mean_probabilities, valid = segment_image(
                segmentation_model,
                processed["rgb"],
                DEVICE,
            )
            metadata = processed["metadata"]
            photo = stretch_to_uint8(processed["rgb"])
            painted = colorize_prediction(class_map)

            pixel_summary = {
                name: int(((class_map == index) & valid).sum())
                for index, name in enumerate(CLASS_NAMES)
            }
            total_pixels = int(valid.sum())
            average_probabilities = {
                name: round(float(mean_probabilities[index]) * 100, 2)
                for index, name in enumerate(CLASS_NAMES)
            }

            return {
                "success": True,
                "filename": file.filename,
                "image_type": "GeoTIFF",
                "model_name": SEGMENTATION_METADATA["architecture"],
                "preview_image": encode_png(photo),
                "segmentation_map": encode_png(painted),
                "preview_width": metadata["width"],
                "preview_height": metadata["height"],
                "total_pixels": total_pixels,
                "pixel_summary": pixel_summary,
                "mean_probabilities": average_probabilities,
                "summary": pixel_summary,
                "predictions": [],
                "metadata": metadata,
            }

        predictions = predict_patches(
            processed["patches"]
        )

        for prediction, position in zip(
            predictions,
            processed["positions"]
        ):
            prediction["position"] = position

        metadata = processed["metadata"]
        photo, _display_scale = upscale_for_display(
            stretch_to_uint8(processed["rgb"])
        )

        return {
            "success": True,
            "filename": file.filename,
            "image_type": "GeoTIFF",
            "total_patches": len(predictions),
            "patch_size": PATCH_SIZE,
            "skipped_patches": (
                metadata["grid_rows"] * metadata["grid_cols"]
                - len(predictions)
            ),
            "patch_grid": {
                "rows": metadata["grid_rows"],
                "columns": metadata["grid_cols"]
            },
            "preview_image": encode_png(photo),
            "model_name": MODEL_METADATA["architecture"],
            "preview_width": metadata["width"],
            "preview_height": metadata["height"],
            "metadata": metadata,
            "predictions": predictions,
            "summary": create_summary(predictions)
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
