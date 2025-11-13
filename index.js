import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

// === CONFIGURACIÓN ===
const HF_TOKEN = process.env.HF_TOKEN;
const MODEL = "intfloat/multilingual-e5-large";
const API_URL = `https://router.huggingface.co/hf-inference/models/${MODEL}/pipeline/feature-extraction`;

// === RUTA PRINCIPAL ===
app.post("/embed", async (req, res) => {
  try {
    const { texto } = req.body;
    if (!texto || texto.trim().length < 5) {
      return res.status(400).json({ error: "Texto vacío o demasiado corto" });
    }

    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: texto }),
    });

    const data = await response.json();

    // Validar formato de respuesta
    if (!Array.isArray(data)) {
      console.error("❌ Formato inesperado:", data);
      return res.status(500).json({ error: "Formato inesperado", data });
    }

    // Aplanar si es matriz de embeddings por token
    let embedding = data;
    if (Array.isArray(data[0])) {
      const tokens = data;
      const dim = tokens[0].length;
      embedding = new Array(dim).fill(0);
      for (const tokenVec of tokens) {
        for (let i = 0; i < dim; i++) embedding[i] += tokenVec[i];
      }
      embedding = embedding.map(v => v / tokens.length);
    }

    res.json({ embedding, dim: embedding.length });
  } catch (err) {
    console.error("❌ Error general:", err);
    res.status(500).json({ error: err.message });
  }
});

// === RUTA DE PRUEBA ===
app.get("/", (req, res) => {
  res.send("✅ RAG Embedding Service activo (Hugging Face)");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Servidor RAG activo en puerto ${PORT}`));
