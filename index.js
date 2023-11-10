import { pipeline } from "@xenova/transformers";
import wavefile from "wavefile";
import fs from "fs";

const EMBED =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin";
const PHRASE = "Hello World!";

const synthesizer = await pipeline("text-to-speech", "Xenova/speecht5_tts", {
  quantized: false,
});

const out = await synthesizer(PHRASE, {
  speaker_embeddings: EMBED,
});

const wav = new wavefile.WaveFile();
wav.fromScratch(1, out.sampling_rate, "32f", out.audio);
fs.writeFileSync("out.wav", wav.toBuffer());
