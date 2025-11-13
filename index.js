import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();
app.use(cors());

// *** ESTA LÍNEA ES LO QUE FALTABA ***
app.use(express.json({ limit: "10mb" }));

const HF_TOKEN = process.env.HF_TOKEN;
const MODEL = "intfloat/multilingual-e5-large";

app.post("/embed", async (req, res) => {
    try {
        const text = req.body.text;

        if (!text || text.trim().length < 10) {
            return res.status(400).json({ error: "Texto vacío o demasiado corto" });
        }

        const payload = {
            model: MODEL,
            provider: "hf-inference",
            inputs: `query: ${text}`
        };

        const response = await fetch(
            `https://router.huggingface.co/hf-inference/models/${MODEL}/pipeline/feature-extraction`,
            {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${HF_TOKEN}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            }
        );

        const result = await response.json();

        if (!Array.isArray(result) || !Array.isArray(result[0])) {
            console.error("Respuesta inesperada:", result);
            return res.status(500).json({ error: "Respuesta inesperada de HuggingFace", raw: result });
        }

        return res.json({ embedding: result[0] });

    } catch (e) {
        console.error("ERROR /embed:", e);
        res.status(500).json({ error: "Error procesando embedding" });
    }
});

const port = process.env.PORT || 10000;
app.listen(port, () => console.log(`RAG service listening on port ${port}`));
