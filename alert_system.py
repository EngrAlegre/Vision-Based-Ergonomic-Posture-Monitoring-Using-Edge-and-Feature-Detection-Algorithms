import time
import platform

class AlertSystem:
    """
    Cross-platform alert system for poor posture warnings.
    Provides visual and optional audio feedback.
    """
    
    def __init__(self, enable_sound=True, alert_cooldown=5.0):
        """
        Initialize alert system.
        
        Args:
            enable_sound: Whether to play audio alerts
            alert_cooldown: Minimum seconds between audio alerts
        """
        self.enable_sound = enable_sound
        self.alert_cooldown = alert_cooldown
        self.last_alert_time = 0
        
        # Detect platform for audio support
        self.system = platform.system()
        self.audio_available = False
        
        if self.enable_sound:
            self._initialize_audio()
    
    def _initialize_audio(self):
        """Initialize audio system based on platform."""
        try:
            if self.system == "Windows":
                import winsound
                self.audio_available = True
                self.play_sound = self._play_sound_windows
            elif self.system == "Darwin":  # macOS
                self.audio_available = True
                self.play_sound = self._play_sound_mac
            elif self.system == "Linux":
                self.audio_available = True
                self.play_sound = self._play_sound_linux
        except ImportError:
            self.audio_available = False
            print("⚠️ Audio alerts not available on this system")
    
    def _play_sound_windows(self):
        """Play alert sound on Windows."""
        try:
            import winsound
            winsound.Beep(900, 250)  # 900 Hz for 250ms
        except Exception:
            pass
    
    def _play_sound_mac(self):
        """Play alert sound on macOS."""
        try:
            import os
            os.system('afplay /System/Library/Sounds/Pop.aiff')
        except Exception:
            pass
    
    def _play_sound_linux(self):
        """Play alert sound on Linux."""
        try:
            import os
            os.system('paplay /usr/share/sounds/freedesktop/stereo/bell.oga &')
        except Exception:
            pass
    
    def should_play_sound(self):
        """
        Check if enough time has passed to play another alert.
        
        Returns:
            Boolean indicating if sound should play
        """
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.last_alert_time = current_time
            return True
        return False
    
    def trigger_alert(self, analysis_result):
        """
        Trigger alert if posture is poor.
        
        Args:
            analysis_result: Dictionary from PostureAnalyzer
            
        Returns:
            Boolean indicating if alert was triggered
        """
        is_poor = analysis_result.get('is_poor_posture', False)
        
        if is_poor and self.audio_available and self.should_play_sound():
            self.play_sound()
            return True
        
        return False
    
    def get_alert_message(self, analysis_result):
        """
        Generate appropriate alert message based on analysis.
        
        Args:
            analysis_result: Dictionary from PostureAnalyzer
            
        Returns:
            String with alert message
        """
        issues = analysis_result.get('issues', [])
        score = analysis_result.get('score', 100)
        
        if score >= 70:
            return "✓ Maintain good posture!"
        elif score >= 50:
            return "⚠ Posture needs attention"
        else:
            return "❌ Poor posture detected!"
    
    def reset_cooldown(self):
        """Reset alert cooldown timer."""
        self.last_alert_time = 0
