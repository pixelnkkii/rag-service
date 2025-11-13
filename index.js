import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

// === CONFIGURACIÓN ===
const HF_TOKEN = process.env.HF_TOKEN;
const MODEL = "intfloat/multilingual-e5-large";

// === RUTA PRINCIPAL ===
app.post("/embed", async (req, res) => {
  try {
    const { texto } = req.body;
    if (!texto || texto.trim().length < 5) {
      return res.status(400).json({ error: "Texto inválido o muy corto" });
    }

    const response = await fetch(`https://router.huggingface.co/hf-inference/models/${MODEL}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: [`query: ${texto}`],
        task: "feature-extraction",
        options: { wait_for_model: true },
      }),
    });

    const data = await response.json();
    if (!Array.isArray(data)) {
      return res.status(500).json({ error: "Formato inesperado", data });
    }

    const embedding = data[0].map((v) => parseFloat(v.toFixed(6)));
    res.json({ embedding, dim: embedding.length });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get("/", (req, res) => {
  res.send("RAG Service activo");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Servidor RAG corriendo en puerto ${PORT}`));

