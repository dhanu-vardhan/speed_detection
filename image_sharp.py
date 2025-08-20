import cv2
import numpy as np
import os
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import shutil

# Enhanced imports for advanced image processing
try:
    from skimage import filters, measure, restoration, exposure, transform
    from skimage.metrics import structural_similarity as ssim
    from scipy import ndimage, signal
    from PIL import Image, ImageEnhance, ImageFilter
    import matplotlib.pyplot as plt
    ADVANCED_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced libraries not available: {e}")
    ADVANCED_LIBS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """Enumeration of different quality metrics"""
    LAPLACIAN_VARIANCE = "laplacian_variance"
    SOBEL_GRADIENT = "sobel_gradient" 
    BRENNER_FOCUS = "brenner_focus"
    TENENGRAD = "tenengrad"
    VARIANCE_OF_LAPLACIAN = "variance_of_laplacian"
    FFT_FREQUENCY = "fft_frequency"
    CANNY_EDGES = "canny_edges"
    SSIM_QUALITY = "ssim_quality"

@dataclass
class QualityScore:
    """Container for image quality metrics"""
    combined_score: float
    individual_scores: Dict[str, float]
    blur_detected: bool
    enhancement_applied: bool
    processing_time: float
    image_size: Tuple[int, int]
    
class AdvancedImageQualityAssessor:
    """Advanced image quality assessment using multiple metrics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.reference_image = None
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'min_sharpness_threshold': 15.0,
            'blur_threshold': 0.15,
            'enhancement_enabled': True,
            'upscale_factor': 1.5,
            'quality_weights': {
                QualityMetric.LAPLACIAN_VARIANCE.value: 0.25,
                QualityMetric.SOBEL_GRADIENT.value: 0.20,
                QualityMetric.BRENNER_FOCUS.value: 0.15,
                QualityMetric.TENENGRAD.value: 0.15,
                QualityMetric.FFT_FREQUENCY.value: 0.15,
                QualityMetric.CANNY_EDGES.value: 0.10
            }
        }
    
    def assess_image_quality(self, image: np.ndarray) -> QualityScore:
        """Comprehensive image quality assessment"""
        start_time = datetime.now()
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate individual quality metrics
        scores = {}
        
        # 1. Laplacian Variance (Primary sharpness measure)
        scores[QualityMetric.LAPLACIAN_VARIANCE.value] = self._laplacian_variance(gray)
        
        # 2. Sobel Gradient Magnitude
        scores[QualityMetric.SOBEL_GRADIENT.value] = self._sobel_gradient(gray)
        
        # 3. Brenner Gradient Focus Measure
        scores[QualityMetric.BRENNER_FOCUS.value] = self._brenner_focus(gray)
        
        # 4. Tenengrad (Gradient-based focus measure)
        scores[QualityMetric.TENENGRAD.value] = self._tenengrad_focus(gray)
        
        # 5. FFT Frequency Analysis
        scores[QualityMetric.FFT_FREQUENCY.value] = self._fft_frequency_analysis(gray)
        
        # 6. Canny Edge Density
        scores[QualityMetric.CANNY_EDGES.value] = self._canny_edge_density(gray)
        
        # Advanced metrics if libraries available
        if ADVANCED_LIBS_AVAILABLE:
            scores[QualityMetric.VARIANCE_OF_LAPLACIAN.value] = self._skimage_variance_laplacian(gray)
        
        # Calculate combined weighted score
        combined_score = self._calculate_weighted_score(scores)
        
        # Detect blur
        blur_detected = self._detect_blur_comprehensive(gray, scores)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QualityScore(
            combined_score=combined_score,
            individual_scores=scores,
            blur_detected=blur_detected,
            enhancement_applied=False,
            processing_time=processing_time,
            image_size=image.shape[:2]
        )
    
    def _laplacian_variance(self, gray: np.ndarray) -> float:
        """Calculate Laplacian variance for sharpness"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _sobel_gradient(self, gray: np.ndarray) -> float:
        """Calculate Sobel gradient magnitude"""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return float(np.mean(magnitude))
    
    def _brenner_focus(self, gray: np.ndarray) -> float:
        """Calculate Brenner gradient focus measure"""
        if gray.shape[0] < 2:
            return 0.0
        
        diff_y = np.diff(gray, axis=0)
        if diff_y.shape[0] < 1:
            return 0.0
            
        brenner = np.sum((diff_y[:-1, :] * diff_y[1:, :])**2)
        return float(brenner / max(gray.shape[0] * gray.shape[1], 1))
    
    def _tenengrad_focus(self, gray: np.ndarray) -> float:
        """Calculate Tenengrad focus measure using Sobel operator"""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = sobelx**2 + sobely**2
        return float(np.mean(tenengrad))
    
    def _fft_frequency_analysis(self, gray: np.ndarray) -> float:
        """Analyze frequency content using FFT"""
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create high-frequency mask
        mask_size = min(rows, cols) // 8
        mask = np.zeros((rows, cols), dtype=np.float32)
        mask[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 1
        
        # Calculate high-frequency energy ratio
        total_energy = np.sum(magnitude_spectrum)
        high_freq_energy = np.sum(magnitude_spectrum * (1 - mask))
        
        if total_energy > 0:
            return float(high_freq_energy / total_energy)
        return 0.0
    
    def _canny_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density using Canny edge detector"""
        # Adaptive thresholds based on image statistics
        sigma = 0.33
        median_val = np.median(gray)
        lower_thresh = int(max(0, (1.0 - sigma) * median_val))
        upper_thresh = int(min(255, (1.0 + sigma) * median_val))
        
        edges = cv2.Canny(gray, lower_thresh, upper_thresh)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        return float(edge_density * 1000)  # Scale for better weighting
    
    def _skimage_variance_laplacian(self, gray: np.ndarray) -> float:
        """Advanced Laplacian variance using scikit-image"""
        if not ADVANCED_LIBS_AVAILABLE:
            return 0.0
        
        laplacian = filters.laplace(gray)
        return float(np.var(laplacian))
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted combination of quality scores"""
        weights = self.config['quality_weights']
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            if metric in weights:
                total_score += weights[metric] * score
                total_weight += weights[metric]
        
        return total_score / max(total_weight, 1.0)
    
    def _detect_blur_comprehensive(self, gray: np.ndarray, scores: Dict[str, float]) -> bool:
        """Comprehensive blur detection using multiple methods"""
        # Primary blur detection using Laplacian variance
        laplacian_blur = scores.get(QualityMetric.LAPLACIAN_VARIANCE.value, 0) < 50
        
        # FFT-based blur detection
        fft_blur = scores.get(QualityMetric.FFT_FREQUENCY.value, 0) < self.config['blur_threshold']
        
        # Edge density blur detection
        edge_blur = scores.get(QualityMetric.CANNY_EDGES.value, 0) < 2.0
        
        # Combined blur decision (any two methods agree)
        blur_votes = sum([laplacian_blur, fft_blur, edge_blur])
        return blur_votes >= 2

class AdvancedImageEnhancer:
    """Advanced image enhancement using multiple techniques"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'upscale_factor': 1.5,
            'upscale_method': 'lanczos',
            'deblur_enabled': True,
            'noise_reduction_enabled': True,
            'contrast_enhancement_enabled': True,
            'gamma_correction': 1.2,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8),
            'unsharp_mask_strength': 0.5,
            'bilateral_filter_d': 9,
            'bilateral_filter_sigma_color': 75,
            'bilateral_filter_sigma_space': 75
        }
    
    def enhance_image(self, image: np.ndarray, quality_score: QualityScore) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Comprehensive image enhancement pipeline"""
        enhanced = image.copy()
        enhancement_log = {'steps': [], 'improvements': {}}
        
        # 1. Upscaling for better resolution
        if self.config['upscale_factor'] > 1.0:
            enhanced = self._upscale_image(enhanced)
            enhancement_log['steps'].append('upscaling')
        
        # 2. Noise reduction
        if self.config['noise_reduction_enabled']:
            enhanced = self._reduce_noise(enhanced)
            enhancement_log['steps'].append('noise_reduction')
        
        # 3. Deblurring if blur detected
        if quality_score.blur_detected and self.config['deblur_enabled']:
            enhanced = self._deblur_image(enhanced)
            enhancement_log['steps'].append('deblurring')
        
        # 4. Contrast enhancement
        if self.config['contrast_enhancement_enabled']:
            enhanced = self._enhance_contrast(enhanced)
            enhancement_log['steps'].append('contrast_enhancement')
        
        # 5. Sharpening
        enhanced = self._apply_unsharp_mask(enhanced)
        enhancement_log['steps'].append('sharpening')
        
        # 6. Gamma correction for visibility
        enhanced = self._apply_gamma_correction(enhanced)
        enhancement_log['steps'].append('gamma_correction')
        
        # 7. Final quality check and adjustment
        enhanced = self._final_quality_adjustment(enhanced)
        enhancement_log['steps'].append('final_adjustment')
        
        return enhanced, enhancement_log
    
    def _upscale_image(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using advanced interpolation"""
        height, width = image.shape[:2]
        new_height = int(height * self.config['upscale_factor'])
        new_width = int(width * self.config['upscale_factor'])
        
        method = self.config['upscale_method'].lower()
        
        if method == 'lanczos':
            interpolation = cv2.INTER_LANCZOS4
        elif method == 'cubic':
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_LINEAR
        
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Advanced noise reduction using bilateral filtering"""
        return cv2.bilateralFilter(
            image,
            d=self.config['bilateral_filter_d'],
            sigmaColor=self.config['bilateral_filter_sigma_color'],
            sigmaSpace=self.config['bilateral_filter_sigma_space']
        )
    
    def _deblur_image(self, image: np.ndarray) -> np.ndarray:
        """Advanced deblurring using unsharp masking and Richardson-Lucy if available"""
        # Primary deblurring with unsharp mask
        gaussian_blur = cv2.GaussianBlur(image, (0, 0), 1.5)
        unsharp_masked = cv2.addWeighted(image, 1.8, gaussian_blur, -0.8, 0)
        
        # Advanced deblurring with scikit-image if available
        if ADVANCED_LIBS_AVAILABLE:
            try:
                # Convert to float for processing
                image_float = image.astype(np.float64) / 255.0
                
                # Create a simple PSF (point spread function) for deconvolution
                psf = np.ones((5, 5)) / 25
                
                # Richardson-Lucy deconvolution
                if len(image.shape) == 3:
                    deconvolved = np.zeros_like(image_float)
                    for i in range(3):
                        deconvolved[:, :, i] = restoration.richardson_lucy(
                            image_float[:, :, i], psf, num_iter=5
                        )
                else:
                    deconvolved = restoration.richardson_lucy(image_float, psf, num_iter=5)
                
                # Combine unsharp mask with deconvolution
                deconvolved = (deconvolved * 255).astype(np.uint8)
                return cv2.addWeighted(unsharp_masked, 0.7, deconvolved, 0.3, 0)
                
            except Exception as e:
                logger.warning(f"Advanced deblurring failed: {e}, using unsharp mask only")
        
        return unsharp_masked
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Advanced contrast enhancement using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_grid_size']
            )
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_grid_size']
            )
            return clahe.apply(image)
    
    def _apply_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for sharpening"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
        strength = self.config['unsharp_mask_strength']
        return cv2.addWeighted(image, 1 + strength, gaussian, -strength, 0)
    
    def _apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction for better visibility"""
        gamma = self.config['gamma_correction']
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def _final_quality_adjustment(self, image: np.ndarray) -> np.ndarray:
        """Final quality adjustments"""
        # Ensure pixel values are in valid range
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Apply slight smoothing to reduce artifacts
        return cv2.bilateralFilter(image, 5, 50, 50)

class VehicleSnapshotManager:
    """Manages vehicle snapshots with quality assessment and enhancement"""
    
    def __init__(self, output_dir: str, config: Dict[str, Any] = None):
        self.output_dir = Path(output_dir)
        self.config = config or self._default_config()
        
        # Initialize components
        self.quality_assessor = AdvancedImageQualityAssessor(config)
        self.image_enhancer = AdvancedImageEnhancer(config)
        
        # Create directory structure
        self._setup_directories()
        
        # Snapshot storage
        self.vehicle_snapshots = {}
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_snapshots_per_vehicle': 5,
            'min_sharpness_threshold': 15.0,
            'min_frames_between_snapshots': 5,
            'save_quality': 95,
            'save_original': True,
            'save_enhanced': True,
            'save_metadata': True,
            'enhancement_enabled': True
        }
    
    def _setup_directories(self):
        """Create directory structure for snapshots"""
        directories = [
            self.output_dir,
            self.output_dir / 'snapshots' / 'original',
            self.output_dir / 'snapshots' / 'enhanced',
            self.output_dir / 'best_snapshots' / 'original',
            self.output_dir / 'best_snapshots' / 'enhanced',
            self.output_dir / 'overspeed' / 'original',
            self.output_dir / 'overspeed' / 'enhanced',
            self.output_dir / 'metadata'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_vehicle_detection(self, 
                                vehicle_id: int, 
                                frame_num: int,
                                bbox: Tuple[int, int, int, int],
                                frame: np.ndarray,
                                speed_kmph: float,
                                class_name: str,
                                is_overspeed: bool = False) -> Optional[Dict[str, Any]]:
        """Process a vehicle detection and potentially capture snapshot"""
        
        # Initialize vehicle data if needed
        if vehicle_id not in self.vehicle_snapshots:
            self.vehicle_snapshots[vehicle_id] = {
                'snapshots': [],
                'best_snapshot': None,
                'last_snapshot_frame': 0,
                'class_name': class_name,
                'total_detections': 0
            }
        
        vehicle_data = self.vehicle_snapshots[vehicle_id]
        vehicle_data['total_detections'] += 1
        
        # Check if we should capture a snapshot
        if not self._should_capture_snapshot(vehicle_id, frame_num):
            return None
        
        # Extract vehicle crop with padding
        vehicle_crop = self._extract_vehicle_crop(bbox, frame)
        if vehicle_crop is None:
            return None
        
        # Assess image quality
        quality_score = self.quality_assessor.assess_image_quality(vehicle_crop)
        
        # Skip if quality is too low
        if quality_score.combined_score < self.config['min_sharpness_threshold']:
            logger.info(f"Vehicle {vehicle_id}: Quality too low ({quality_score.combined_score:.1f})")
            return None
        
        # Enhance image if enabled
        enhanced_crop = None
        enhancement_log = None
        if self.config['enhancement_enabled']:
            enhanced_crop, enhancement_log = self.image_enhancer.enhance_image(vehicle_crop, quality_score)
        
        # Save snapshots
        snapshot_data = self._save_snapshots(
            vehicle_id, frame_num, vehicle_crop, enhanced_crop,
            quality_score, speed_kmph, class_name, is_overspeed, enhancement_log
        )
        
        # Update vehicle data
        vehicle_data['snapshots'].append(snapshot_data)
        vehicle_data['last_snapshot_frame'] = frame_num
        
        # Update best snapshot if this one is better
        self._update_best_snapshot(vehicle_id, snapshot_data)
        
        logger.info(f"Snapshot captured for Vehicle {vehicle_id}: {speed_kmph:.1f} km/h "
                   f"(Quality: {quality_score.combined_score:.1f})")
        
        return snapshot_data
    
    def _should_capture_snapshot(self, vehicle_id: int, frame_num: int) -> bool:
        """Determine if we should capture a snapshot"""
        vehicle_data = self.vehicle_snapshots[vehicle_id]
        
        # Check max snapshots limit
        if len(vehicle_data['snapshots']) >= self.config['max_snapshots_per_vehicle']:
            return False
        
        # Check minimum frames between snapshots
        frames_since_last = frame_num - vehicle_data['last_snapshot_frame']
        if frames_since_last < self.config['min_frames_between_snapshots']:
            return False
        
        return True
    
    def _extract_vehicle_crop(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract vehicle crop from frame with validation"""
        x1, y1, x2, y2 = bbox
        
        # Add padding
        padding_x = int((x2 - x1) * 0.15)
        padding_y = int((y2 - y1) * 0.15)
        
        # Apply bounds checking
        height, width = frame.shape[:2]
        crop_x1 = max(0, x1 - padding_x)
        crop_y1 = max(0, y1 - padding_y)
        crop_x2 = min(width, x2 + padding_x)
        crop_y2 = min(height, y2 + padding_y)
        
        # Extract crop
        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Validate crop
        if not self._validate_crop(vehicle_crop):
            return None
        
        return vehicle_crop
    
    def _validate_crop(self, crop: np.ndarray, min_size: int = 50) -> bool:
        """Validate vehicle crop quality"""
        if crop.size == 0:
            return False
        
        height, width = crop.shape[:2]
        
        # Size validation
        if width < min_size or height < min_size:
            return False
        
        if width > 2000 or height > 2000:
            return False
        
        # Aspect ratio validation
        aspect_ratio = height / width if width > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            return False
        
        # Contrast validation
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        if np.std(gray) < 10:
            return False
        
        return True
    
    def _save_snapshots(self, 
                       vehicle_id: int, 
                       frame_num: int,
                       original_crop: np.ndarray,
                       enhanced_crop: Optional[np.ndarray],
                       quality_score: QualityScore,
                       speed_kmph: float,
                       class_name: str,
                       is_overspeed: bool,
                       enhancement_log: Optional[Dict]) -> Dict[str, Any]:
        """Save snapshot images and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        speed_str = f"{int(speed_kmph)}kmph"
        base_filename = f"vehicle_{vehicle_id}_{class_name}_{speed_str}_{timestamp}"
        
        # Determine subdirectory
        subdir = "overspeed" if is_overspeed else "snapshots"
        
        snapshot_data = {
            'vehicle_id': vehicle_id,
            'frame_num': frame_num,
            'timestamp': datetime.now().isoformat(),
            'speed_kmph': speed_kmph,
            'class_name': class_name,
            'is_overspeed': is_overspeed,
            'quality_score': quality_score.combined_score,
            'individual_scores': quality_score.individual_scores,
            'blur_detected': quality_score.blur_detected,
            'processing_time': quality_score.processing_time,
            'image_size': quality_score.image_size,
            'enhancement_log': enhancement_log,
            'paths': {}
        }
        
        # Save original image
        if self.config['save_original']:
            original_path = self.output_dir / subdir / 'original' / f"{base_filename}_original.jpg"
            cv2.imwrite(str(original_path), original_crop, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.config['save_quality']])
            snapshot_data['paths']['original'] = str(original_path)
        
        # Save enhanced image
        if self.config['save_enhanced'] and enhanced_crop is not None:
            enhanced_path = self.output_dir / subdir / 'enhanced' / f"{base_filename}_enhanced.jpg"
            cv2.imwrite(str(enhanced_path), enhanced_crop,
                       [cv2.IMWRITE_JPEG_QUALITY, self.config['save_quality']])
            snapshot_data['paths']['enhanced'] = str(enhanced_path)
        
        # Save metadata
        if self.config['save_metadata']:
            metadata_path = self.output_dir / 'metadata' / f"{base_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)
        
        return snapshot_data
    
    def _update_best_snapshot(self, vehicle_id: int, snapshot_data: Dict[str, Any]):
        """Update best snapshot for vehicle based on quality score"""
        current_best = self.vehicle_snapshots[vehicle_id]['best_snapshot']
        
        if (current_best is None or 
            snapshot_data['quality_score'] > current_best['quality_score']):
            self.vehicle_snapshots[vehicle_id]['best_snapshot'] = snapshot_data.copy()
    
    def create_best_snapshots_collection(self):
        """Create collection of best snapshots for each vehicle"""
        logger.info("Creating best snapshots collection...")
        
        for vehicle_id, vehicle_data in self.vehicle_snapshots.items():
            best_snapshot = vehicle_data.get('best_snapshot')
            if not best_snapshot:
                continue
            
            # Copy best snapshots to dedicated folder
            for image_type in ['original', 'enhanced']:
                src_path = best_snapshot['paths'].get(image_type)
                if not src_path or not os.path.exists(src_path):
                    continue
                
                # Determine destination
                if best_snapshot['is_overspeed']:
                    dst_dir = self.output_dir / 'overspeed' / image_type
                else:
                    dst_dir = self.output_dir / 'best_snapshots' / image_type
                
                dst_path = dst_dir / f"best_{os.path.basename(src_path)}"
                shutil.copy2(src_path, dst_path)
                
                best_snapshot['paths'][f'best_{image_type}'] = str(dst_path)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        total_vehicles = len(self.vehicle_snapshots)
        total_snapshots = sum(len(v['snapshots']) for v in self.vehicle_snapshots.values())
        vehicles_with_snapshots = len([v for v in self.vehicle_snapshots.values() if v['snapshots']])
        
        # Quality statistics
        all_quality_scores = []
        blur_count = 0
        enhancement_count = 0
        
        for vehicle_data in self.vehicle_snapshots.values():
            for snapshot in vehicle_data['snapshots']:
                all_quality_scores.append(snapshot['quality_score'])
                if snapshot['blur_detected']:
                    blur_count += 1
                if snapshot.get('enhancement_log'):
                    enhancement_count += 1
        
        # Class breakdown
        class_stats = {}
        for vehicle_data in self.vehicle_snapshots.values():
            class_name = vehicle_data['class_name']
            if class_name not in class_stats:
                class_stats[class_name] = {
                    'count': 0,
                    'snapshots': 0,
                    'avg_quality': 0.0,
                    'best_quality': 0.0
                }
            
            class_stats[class_name]['count'] += 1
            class_stats[class_name]['snapshots'] += len(vehicle_data['snapshots'])
            
            # Quality stats for this class
            class_qualities = [s['quality_score'] for s in vehicle_data['snapshots']]
            if class_qualities:
                class_stats[class_name]['avg_quality'] = np.mean(class_qualities)
                class_stats[class_name]['best_quality'] = max(class_qualities)
        
        # Processing time statistics
        processing_times = []
        for vehicle_data in self.vehicle_snapshots.values():
            for snapshot in vehicle_data['snapshots']:
                processing_times.append(snapshot['processing_time'])
        
        report = {
            'summary': {
                'total_vehicles': total_vehicles,
                'total_snapshots': total_snapshots,
                'vehicles_with_snapshots': vehicles_with_snapshots,
                'snapshot_success_rate': (vehicles_with_snapshots / max(total_vehicles, 1)) * 100,
                'avg_snapshots_per_vehicle': total_snapshots / max(total_vehicles, 1)
            },
            'quality_metrics': {
                'avg_quality_score': np.mean(all_quality_scores) if all_quality_scores else 0,
                'min_quality_score': np.min(all_quality_scores) if all_quality_scores else 0,
                'max_quality_score': np.max(all_quality_scores) if all_quality_scores else 0,
                'quality_std': np.std(all_quality_scores) if all_quality_scores else 0,
                'blur_detection_count': blur_count,
                'enhancement_applied_count': enhancement_count
            },
            'class_breakdown': class_stats,
            'performance': {
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'total_processing_time': np.sum(processing_times) if processing_times else 0
            },
            'directories': {
                'snapshots_original': str(self.output_dir / 'snapshots' / 'original'),
                'snapshots_enhanced': str(self.output_dir / 'snapshots' / 'enhanced'),
                'best_snapshots_original': str(self.output_dir / 'best_snapshots' / 'original'),
                'best_snapshots_enhanced': str(self.output_dir / 'best_snapshots' / 'enhanced'),
                'overspeed_original': str(self.output_dir / 'overspeed' / 'original'),
                'overspeed_enhanced': str(self.output_dir / 'overspeed' / 'enhanced'),
                'metadata': str(self.output_dir / 'metadata')
            }
        }
        
        return report

class OptimizedVehicleSnapshotter:
    """Main class for optimized vehicle snapshot processing"""
    
    def __init__(self, output_dir: str, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.snapshot_manager = VehicleSnapshotManager(output_dir, config)
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            # Quality thresholds
            'min_sharpness_threshold': 15.0,
            'blur_threshold': 0.15,
            
            # Snapshot management
            'max_snapshots_per_vehicle': 3,
            'min_frames_between_snapshots': 5,
            
            # Image processing
            'enhancement_enabled': True,
            'upscale_factor': 1.5,
            'save_quality': 95,
            
            # File management
            'save_original': True,
            'save_enhanced': True,
            'save_metadata': True,
            
            # Speed detection
            'speed_limit_kmph': 40,
            
            # Advanced settings
            'quality_weights': {
                'laplacian_variance': 0.25,
                'sobel_gradient': 0.20,
                'brenner_focus': 0.15,
                'tenengrad': 0.15,
                'fft_frequency': 0.15,
                'canny_edges': 0.10
            }
        }
    
    def process_detection(self, 
                         vehicle_id: int,
                         frame_num: int,
                         bbox: Tuple[int, int, int, int],
                         frame: np.ndarray,
                         speed_kmph: float,
                         class_name: str) -> Optional[Dict[str, Any]]:
        """Process a single vehicle detection"""
        is_overspeed = speed_kmph > self.config['speed_limit_kmph']
        
        return self.snapshot_manager.process_vehicle_detection(
            vehicle_id, frame_num, bbox, frame, speed_kmph, class_name, is_overspeed
        )
    
    def finalize_processing(self) -> Dict[str, Any]:
        """Finalize processing and generate reports"""
        # Create best snapshots collection
        self.snapshot_manager.create_best_snapshots_collection()
        
        # Generate summary report
        report = self.snapshot_manager.generate_summary_report()
        
        # Save report to file
        report_path = Path(self.snapshot_manager.output_dir) / 'processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def get_best_snapshots(self) -> Dict[int, Dict[str, Any]]:
        """Get best snapshots for all vehicles"""
        best_snapshots = {}
        for vehicle_id, vehicle_data in self.snapshot_manager.vehicle_snapshots.items():
            if vehicle_data['best_snapshot']:
                best_snapshots[vehicle_id] = vehicle_data['best_snapshot']
        
        return best_snapshots

# Example usage and testing functions
def demo_usage():
    """Demonstrate usage of the optimized snapshot system"""
    
    # Configuration for the snapshot system
    config = {
        'min_sharpness_threshold': 15.0,
        'max_snapshots_per_vehicle': 3,
        'enhancement_enabled': True,
        'upscale_factor': 1.5,
        'save_quality': 95,
        'speed_limit_kmph': 40
    }
    
    # Initialize the snapshotter
    output_dir = "output_dir = /Users/anthapuvivekanandareddy/Desktop/Speed_detection/output"
    snapshotter = OptimizedVehicleSnapshotter(output_dir, config)
    
    # Example: Process detections from video frames
    """
    # In your main detection loop:
    for frame_num, frame in enumerate(video_frames):
        # Your YOLO detection code here
        detections = yolo_model.detect(frame)
        
        for detection in detections:
            vehicle_id = detection['id']
            bbox = detection['bbox']  # (x1, y1, x2, y2)
            class_name = detection['class_name']
            speed_kmph = calculate_speed(vehicle_id)  # Your speed calculation
            
            # Process the detection
            snapshot_data = snapshotter.process_detection(
                vehicle_id, frame_num, bbox, frame, speed_kmph, class_name
            )
            
            if snapshot_data:
                print(f"Captured snapshot for vehicle {vehicle_id}: "
                      f"Quality {snapshot_data['quality_score']:.1f}")
    
    # Finalize processing
    final_report = snapshotter.finalize_processing()
    print(f"Processing complete: {final_report['summary']}")
    
    # Get best snapshots
    best_snapshots = snapshotter.get_best_snapshots()
    for vehicle_id, snapshot in best_snapshots.items():
        print(f"Best snapshot for vehicle {vehicle_id}: "
              f"Quality {snapshot['quality_score']:.1f}, "
              f"Speed {snapshot['speed_kmph']:.1f} km/h")
    """

def test_quality_assessment():
    """Test the quality assessment system with sample images"""
    
    # Create a test image (you would load real images here)
    test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    # Initialize quality assessor
    assessor = AdvancedImageQualityAssessor()
    
    # Assess quality
    quality_score = assessor.assess_image_quality(test_image)
    
    print("Quality Assessment Results:")
    print(f"Combined Score: {quality_score.combined_score:.2f}")
    print(f"Individual Scores: {quality_score.individual_scores}")
    print(f"Blur Detected: {quality_score.blur_detected}")
    print(f"Processing Time: {quality_score.processing_time:.4f}s")
    
    return quality_score

def test_image_enhancement():
    """Test the image enhancement system"""
    
    # Create a test image
    test_image = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
    
    # Initialize components
    assessor = AdvancedImageQualityAssessor()
    enhancer = AdvancedImageEnhancer()
    
    # Assess original quality
    original_quality = assessor.assess_image_quality(test_image)
    
    # Enhance image
    enhanced_image, enhancement_log = enhancer.enhance_image(test_image, original_quality)
    
    # Assess enhanced quality
    enhanced_quality = assessor.assess_image_quality(enhanced_image)
    
    print("Enhancement Results:")
    print(f"Original Quality: {original_quality.combined_score:.2f}")
    print(f"Enhanced Quality: {enhanced_quality.combined_score:.2f}")
    print(f"Quality Improvement: {enhanced_quality.combined_score - original_quality.combined_score:.2f}")
    print(f"Enhancement Steps: {enhancement_log['steps']}")
    
    return enhanced_image, enhancement_log

# Quality benchmarking and comparison
class QualityBenchmark:
    """Benchmark different quality assessment methods"""
    
    def __init__(self):
        self.assessor = AdvancedImageQualityAssessor()
    
    def benchmark_methods(self, images: List[np.ndarray]) -> Dict[str, List[float]]:
        """Benchmark different quality assessment methods on a set of images"""
        results = {metric.value: [] for metric in QualityMetric}
        
        for image in images:
            quality_score = self.assessor.assess_image_quality(image)
            for metric, score in quality_score.individual_scores.items():
                results[metric].append(score)
        
        return results
    
    def compare_with_ground_truth(self, 
                                images: List[np.ndarray], 
                                ground_truth_scores: List[float]) -> Dict[str, float]:
        """Compare assessment results with ground truth quality scores"""
        
        predicted_scores = []
        for image in images:
            quality_score = self.assessor.assess_image_quality(image)
            predicted_scores.append(quality_score.combined_score)
        
        # Calculate correlation and error metrics
        correlation = np.corrcoef(predicted_scores, ground_truth_scores)[0, 1]
        mae = np.mean(np.abs(np.array(predicted_scores) - np.array(ground_truth_scores)))
        rmse = np.sqrt(np.mean((np.array(predicted_scores) - np.array(ground_truth_scores))**2))
        
        return {
            'correlation': correlation,
            'mae': mae,
            'rmse': rmse,
            'predicted_scores': predicted_scores
        }

# Main execution example
if __name__ == "__main__":
    print("Enhanced Vehicle Snapshot Quality System")
    print("=" * 50)
    
    # Test quality assessment
    print("\n1. Testing Quality Assessment:")
    quality_score = test_quality_assessment()
    
    # Test image enhancement
    print("\n2. Testing Image Enhancement:")
    enhanced_image, enhancement_log = test_image_enhancement()
    
    # Demo configuration
    print("\n3. Demo Configuration:")
    demo_usage()
    
    print("\n" + "=" * 50)
    print("System Features:")
    print("✓ Advanced multi-metric quality assessment")
    print("✓ Comprehensive image enhancement pipeline")
    print("✓ Intelligent blur detection and correction")
    print("✓ Smart snapshot selection and management")
    print("✓ Detailed quality metrics and reporting")
    print("✓ Configurable processing parameters")
    print("✓ Efficient directory structure organization")
    print("✓ JSON metadata for each snapshot")
    print("✓ Best snapshot selection per vehicle")
    print("✓ Overspeed detection and separate storage")
    
    if ADVANCED_LIBS_AVAILABLE:
        print("✓ Advanced scikit-image processing available")
        print("✓ Richardson-Lucy deconvolution for deblurring")
        print("✓ Enhanced frequency analysis capabilities")
    else:
        print("⚠ Advanced libraries not available (install scikit-image, scipy, PIL)")
    
    print("\nReady for integration with YOLO vehicle detection system!")
    
    # Example integration snippet
    print("\n" + "=" * 50)
    print("Integration Example:")
