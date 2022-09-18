import sys
sys.path.append('tortoise-tts')
import os
import shutil

from tortoise.api import *
from tortoise.utils.audio import *
from sanscript import transliterate, DEVANAGARI, HK

import glob


speakers = {}
texts = {}
top_k=3; num_autoregressive_samples=16; temperature=.8; length_penalty=1; repetition_penalty=2.0; top_p=.8; max_mel_tokens=500;
clvp_cvvp_slider=.5
diffusion_iterations=80; cond_free=True; cond_free_k=2; diffusion_temperature=1.0

with open('train/transcription.txt', 'r') as f:
    for i, line in enumerate(f):
        # print(line)
        line = line.rstrip()
        meta = line.split(' ')[0]
        text = ' '.join(line.split(' ')[1:])
        
        line_id, speaker_id = meta.split('_')
        if speaker_id in speakers:
            speakers[speaker_id].append(line_id)
        else:
            speakers[speaker_id] = [line_id]
        
        texts[line_id] = text
            

wav_files = sorted(glob.glob('train/audio/*.wav'))
wav_files_and_speaker = [(f.split('/')[-1].split('_')[-1][:-4], f) for f in wav_files]
wav_files_per_speaker = {}
for speaker_id, f in wav_files_and_speaker:
    if speaker_id in wav_files_per_speaker:
        wav_files_per_speaker[speaker_id].append(f)
    else:
        wav_files_per_speaker[speaker_id] = [f]
            

@torch.no_grad()
def make_mel(lines, auto_conditioning, diffusion_conditioning, auto_conds, diffuser):
    mels, text_lats = [], []
    for text in tqdm(lines):
        text = transliterate(text, DEVANAGARI, HK).lower()
        text_tokens = torch.IntTensor(tts.tokenizer.encode(text)).unsqueeze(0).cuda()
        text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
        text_lats.append(tts.autoregressive.text_embedding(text_tokens.clone().cpu()))

        samples = []
        num_batches = num_autoregressive_samples // tts.autoregressive_batch_size
        stop_mel_token = tts.autoregressive.stop_mel_token
        calm_token = 83  # This is the token for coding silence, which is fixed in place with "fix_autoregressive_output"
        tts.autoregressive = tts.autoregressive.cuda()
        for b in tqdm(range(num_batches), disable=True):
            codes = tts.autoregressive.inference_speech(auto_conditioning, text_tokens,
                                                            do_sample=True,
                                                            top_p=top_p,
                                                            temperature=temperature,
                                                            num_return_sequences=tts.autoregressive_batch_size,
                                                            length_penalty=length_penalty,
                                                            repetition_penalty=repetition_penalty,
                                                            max_generate_length=max_mel_tokens)
            padding_needed = max_mel_tokens - codes.shape[1]
            codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
            samples.append(codes)
        tts.autoregressive = tts.autoregressive.cpu()

        clip_results = []
        tts.clvp = tts.clvp.cuda()
        tts.cvvp = tts.cvvp.cuda()
        for batch in tqdm(samples, disable=True):
            for i in range(batch.shape[0]):
                batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
            clvp = tts.clvp(text_tokens.repeat(batch.shape[0], 1), batch, return_loss=False)
            if auto_conds is not None:
                cvvp_accumulator = 0
                for cl in range(auto_conds.shape[1]):
                    cvvp_accumulator = cvvp_accumulator + tts.cvvp(auto_conds[:, cl].repeat(batch.shape[0], 1, 1), batch,
                                                                   return_loss=False)
                cvvp = cvvp_accumulator / auto_conds.shape[1]
                clip_results.append(clvp * clvp_cvvp_slider + cvvp * (1-clvp_cvvp_slider))
            else:
                clip_results.append(clvp)
        clip_results = torch.cat(clip_results, dim=0)
        samples = torch.cat(samples, dim=0)
        best_results = samples[torch.topk(clip_results, k=top_k).indices]
        tts.clvp = tts.clvp.cpu()
        tts.cvvp = tts.cvvp.cpu()
        del samples

        # The diffusion model actually wants the last hidden layer from the autoregressive model as conditioning
        # inputs. Re-produce those for the top results. This could be made more efficient by storing all of these
        # results, but will increase memory usage.
        tts.autoregressive = tts.autoregressive.cuda()
        best_latents = tts.autoregressive(auto_conditioning.repeat(top_k, 1), text_tokens.repeat(top_k, 1),
                                            torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), best_results,
                                            torch.tensor([best_results.shape[-1]*tts.autoregressive.mel_length_compression],
                                                         device=text_tokens.device),
                                            return_latent=True, clip_inputs=False)
        tts.autoregressive = tts.autoregressive.cpu()

        tts.diffusion = tts.diffusion.cuda()
        mels_per_line = []
        for b in range(best_results.shape[0]):
            codes = best_results[b].unsqueeze(0)
            latents = best_latents[b].unsqueeze(0)

            # Find the first occurrence of the "calm" token and trim the codes to that.
            ctokens = 0
            for k in range(codes.shape[-1]):
                if codes[0, k] == calm_token:
                    ctokens += 1
                else:
                    ctokens = 0
                if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                    latents = latents[:, :k]
                    break

            mel = do_spectrogram_diffusion(tts.diffusion, diffuser, latents, diffusion_conditioning,
                                            temperature=diffusion_temperature, verbose=False)
            mels_per_line.append(mel.cpu())
        mels.append(mels_per_line)
        tts.diffusion = tts.diffusion.cpu()
    return mels, text_lats
    

tts = TextToSpeech()
diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free, 
                                          cond_free_k=cond_free_k)

for speaker_count, speaker_id in enumerate(tqdm(speakers.keys())):
    print(speaker_count, speaker_id)
    # if speaker_count < 6:
    #     continue
    if os.path.exists('tortoise/voices/temp_voice_folder'):
        shutil.rmtree('tortoise/voices/temp_voice_folder')
    os.makedirs('tortoise/voices/temp_voice_folder', exist_ok=True)
    
    for i, f in enumerate(wav_files_per_speaker[speaker_id][:3]):
        shutil.copyfile(f, f"tortoise/voices/temp_voice_folder/{i}.wav")
        
    voice_samples, _ = load_voice('temp_voice_folder')
    
    auto_conditioning, diffusion_conditioning, auto_conds, _ = tts.get_conditioning_latents(voice_samples, return_mels=True)
    
    # take 10 lines per speaker
    lines = [texts[l_id] for l_id in speakers[speaker_id][:10]]
    mels, text_lats = make_mel(lines, auto_conditioning, diffusion_conditioning, auto_conds, diffuser)
    # import pdb; pdb.set_trace()
    
    for mels_per_line, text_lat, line_id in zip(mels, text_lats, speakers[speaker_id][:10]):
        try:
            targ_wav = load_audio(f'train/audio/{line_id}_{speaker_id}.wav', 24000)
            targ_mel = wav_to_univnet_mel(targ_wav.cuda()).cpu()
        except:
            continue
            
        for idx, mel in enumerate(mels_per_line):
            torch.save((mel, text_lat, diffusion_conditioning), f'train/mels/inputs/{line_id}_{speaker_id}_{idx}.pth')
            torch.save(targ_mel, f'train/mels/targs/{line_id}_{speaker_id}_{idx}.pth')
    
    # if speaker_count >= 19:
    #     break