import ffmpeg
import os
import shutil
from utils.logger import logger

class AudioProcessor:
    @staticmethod
    def get_ffmpeg_path():
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return shutil.which("ffmpeg") or "ffmpeg"

    @staticmethod
    def add_background_music(video_path, music_path, output_path, volume=0.5):
        """
        Overlays background music onto the video using FFmpeg.
        Loops the audio if it's shorter than the video, and cuts it at the video's end.
        """
        logger.info(f"Adding background music: {music_path} to {video_path}")
        ffmpeg_cmd = AudioProcessor.get_ffmpeg_path()
        if not ffmpeg_cmd:
            logger.error("FFmpeg executable could not be found.")
            return False

        try:
            video = ffmpeg.input(video_path)
            
            # Select audio stream explicitly and loop indefinitely
            audio = ffmpeg.input(music_path, stream_loop=-1)['a'].filter('volume', volume)
            
            # shortest=None ensures the output stops when the shortest stream (the video) ends
            out = ffmpeg.output(
                video['v'], 
                audio, 
                output_path, 
                vcodec='copy', 
                acodec='aac', 
                shortest=None
            )
            
            ffmpeg.run(out, cmd=ffmpeg_cmd, overwrite_output=True, quiet=True)
            logger.info("Background music added successfully.")
            return True
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            logger.error(f"FFmpeg error during audio processing: {err_msg}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {e}")
            return False
