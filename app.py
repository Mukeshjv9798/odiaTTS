import os
import torch
import yaml
import time
import librosa
import librosa.display
import IPython
from IPython.display import Audio
import sys
sys.path.append("/") #Path to parent folder of TTS dir
from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis
from TTS.utils.visual import visualize

def tts(model, text, CONFIG, use_cuda, ap, use_gl, figures=True):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, style_wav=None,
                                                                             truncated=False, use_griffin_lim = True, enable_eos_bos_chars=CONFIG.enable_eos_bos_chars, do_trim_silence = False)
    OUT_FOLDER = "/content/output"  #Path where the audio files will be saved
    os.makedirs(OUT_FOLDER, exist_ok=True)
    file_name = text.replace(" ", "_").replace(".","") + ".wav"
    out_path = os.path.join(OUT_FOLDER, file_name)
    ap.save_wav(waveform, out_path)
    return alignment, mel_postnet_spec, stop_tokens, waveform

# model paths
TTS_MODEL = "/content/ttsmodel/checkpoint_290000.pth.tar"
TTS_CONFIG = "/content/ttsmodel/config.json"

TTS_CONFIG = load_config(TTS_CONFIG)

# Run FLAGs
use_cuda = False
# Set some config fields manually for testing
TTS_CONFIG.windowing = False
TTS_CONFIG.use_forward_attn = True 
# Set the vocoder
use_gl = True # use GL if True
batched_wavernn = False    # use batched wavernn inference if True

speaker_id = None
speakers = []

# load the model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speakers), TTS_CONFIG)

# load the audio processor
ap = AudioProcessor(**TTS_CONFIG.audio)         

# load model state
cp =  torch.load(TTS_MODEL, map_location=torch.device('cpu'))

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()
print(cp['step'])
print(cp['r'])

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])

if use_gl == False:
    vocoder_model = ParallelWaveGANGenerator(**PWGAN_CONFIG["generator_params"])
    vocoder_model.load_state_dict(torch.load(PWGAN_MODEL, map_location="cpu")["model"]["generator"])
    vocoder_model.remove_weight_norm()
    ap_vocoder = AudioProcessorVocoder(**PWGAN_CONFIG['audio'])    
    if use_cuda:
        vocoder_model.cuda()
    vocoder_model.eval();

sentence =  "ସେ ଖ୍ରୀଷ୍ଟୀୟ ଷୋଡ଼ଶ ଶତାବ୍ଦୀର ଶେଷାଂସରେ ଓଡ଼ିଶାର ଭ୍ରମଣ କରିଥିଲେ" #Text Input
align, spec, stop_tokens, wav = tts(model, sentence, TTS_CONFIG, use_cuda, ap, use_gl=use_gl, figures = True)   #Function call for audio synthesis
