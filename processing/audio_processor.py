import ffmpeg
import os
from utils.logger import logger

class AudioProcessor:
    @staticmethod
    def add_background_music(video_path, music_path, output_path, volume=0.5):
        """
        Overlays background music onto the video using FFmpeg.
        Loops the audio if it's shorter than the video, and cuts it at the video's end.
        """
        logger.info(f"Adding background music: {music_path} to {video_path}")
        try:
            video = ffmpeg.input(video_path)
            
            # Use stream_loop=-1 to loop the audio indefinitely
            audio = ffmpeg.input(music_path, stream_loop=-1).filter('volume', volume)
            
            # shortest=None ensures the output stops when the shortest stream (the video) ends
            out = ffmpeg.output(
                video.video, 
                audio.audio, 
                output_path, 
                vcodec='copy', 
                acodec='aac', 
                shortest=None
            )
            
            ffmpeg.run(out, overwrite_output=True, quiet=True)
            logger.info("Background music added successfully.")
            return True
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error during audio processing: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {e}")
            return False
