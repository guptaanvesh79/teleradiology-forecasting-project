"""Smoke-check the installed tools without loading assignment data or calling an LLM."""

import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
IMPORT_NAMES = {"scikit-learn": "sklearn"}


def check_versions():
    assert sys.version_info[:3] == (3, 13, 18), sys.version
    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    for filename in ("requirements.txt", "requirements-dev.txt"):
        for line in (ROOT / filename).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r")):
                continue
            name, expected = line.split("==")
            installed = importlib.metadata.version(name)
            assert installed == expected, (name, installed, expected)
            importlib.import_module(IMPORT_NAMES.get(name, name.replace("-", "_")))
            print(f"  {name}=={installed}", flush=True)


def check_models():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    hours = pd.date_range("2026-06-01", periods=168, freq="h")
    features = pd.DataFrame({"hour": hours.hour, "weekday": hours.day_of_week})
    counts = (10 + hours.hour.to_numpy() + hours.day_of_week.to_numpy()).astype(float)
    point = HistGradientBoostingRegressor(
        loss="poisson", max_iter=10, random_state=0
    ).fit(features, counts)
    prediction = point.predict(features.iloc[:24])
    assert prediction.shape == (24,)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    for quantile in (0.1, 0.5, 0.9):
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, max_iter=10, random_state=0
        ).fit(features, counts)
        prediction = model.predict(features.iloc[:24])
        assert prediction.shape == (24,) and np.isfinite(prediction).all()
    print("PASS: pandas/NumPy processing and point/p10/p50/p90 models", flush=True)


def check_api():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel, Field

    class Count(BaseModel):
        studies: int = Field(ge=0)

    app = FastAPI()

    @app.post("/check", response_model=Count)
    def check(payload: Count):
        return payload

    with TestClient(app) as client:
        response = client.post("/check", json={"studies": 12})
        assert response.status_code == 200, response.text
        assert response.json() == {"studies": 12}
        for invalid in ({"studies": -1}, {"studies": "invalid"}, {}):
            assert client.post("/check", json=invalid).status_code == 422
    print("PASS: FastAPI/HTTPX requests and Pydantic validation", flush=True)


def check_chart(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [10, 15, 12])
    ax.set(xlabel="Hour", ylabel="Studies", title="Environment smoke check")
    output = directory / "smoke-chart.png"
    fig.savefig(output)
    plt.close(fig)
    assert output.stat().st_size > 0
    print("PASS: Matplotlib chart rendering", flush=True)


def check_notebook(directory):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager

    spec = KernelSpecManager().get_kernel_spec("main_env")
    assert Path(spec.argv[0]).resolve() == Path(sys.executable).resolve(), spec.argv
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(
            "import sys\n"
            f"assert sys.executable == {sys.executable!r}\n"
            "assert sys.version_info[:3] == (3, 13, 18)\n"
            "print('main_env kernel OK')"
        )]
    )
    executed = NotebookClient(
        notebook, kernel_name="main_env", timeout=60
    ).execute(cwd=str(directory))
    outputs = executed.cells[0].outputs
    assert any("main_env kernel OK" in output.get("text", "") for output in outputs)
    print("PASS: main_env notebook kernel execution", flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="forecasting-setup-") as temporary:
        directory = Path(temporary)
        # Keep notebook and rendering caches inside the temporary directory.
        for variable, name in (
            ("MPLCONFIGDIR", "matplotlib"),
            ("IPYTHONDIR", "ipython"),
            ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
        ):
            cache = directory / name
            cache.mkdir()
            os.environ[variable] = str(cache)
        check_versions()
        check_models()
        check_api()
        check_chart(directory)
        check_notebook(directory)
    print("All Python environment checks passed.", flush=True)


if __name__ == "__main__":
    main()
