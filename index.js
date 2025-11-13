



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
    // aceptar texto o inputs
    let texto = req.body.texto || req.body.inputs;

    if (!texto || typeof texto !== "string" || texto.trim().length < 5) {
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

    // Validar formatos válidos
    if (data.embeddings) {
      return res.json({
        embedding: data.embeddings[0],
        dim: data.embeddings[0].length,
      });
    }

    if (Array.isArray(data) && Array.isArray(data[0])) {
      // token-level → promedio
      const tokens = data;
      const dim = tokens[0].length;
      let avg = new Array(dim).fill(0);

      for (const t of tokens) {
        for (let i = 0; i < dim; i++) avg[i] += t[i];
      }

      avg = avg.map((v) => v / tokens.length);

      return res.json({ embedding: avg, dim });
    }

    return res.status(500).json({ error: "Formato inesperado", data });
  } catch (err) {
    console.error("❌ Error general:", err);
    res.status(500).json({ error: err.message });
  }
});

// === RUTA DE PRUEBA ===
app.get("/", (req, res) => {
  res.send("RAG Embedding Service activo");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Servidor RAG activo en puerto ${PORT}`));

