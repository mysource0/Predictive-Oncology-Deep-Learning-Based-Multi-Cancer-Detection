# Key Loading Animation CSS Details

## DNA Helix Loader Animation

### Main Animation: `dnaHelix`
```css
@keyframes dnaHelix {
    0% {
        transform: translateY(0) scale(0.8);
        opacity: 0.4;
        filter: blur(0.5px);
    }
    35% {
        transform: translateY(-30px) scale(1.15);
        opacity: 1;
        filter: blur(0);
    }
    50% {
        transform: translateY(-38px) scale(1.2);
        opacity: 1;
        filter: blur(0);
    }
    65% {
        transform: translateY(-30px) scale(1.15);
        opacity: 1;
        filter: blur(0);
    }
    100% {
        transform: translateY(0) scale(0.8);
        opacity: 0.4;
        filter: blur(0.5px);
    }
}
```

**Features:**
- 2s duration with cubic-bezier timing
- Vertical wave motion (0 → -38px → 0)
- Scale variation (0.8 → 1.2 → 0.8)
- Opacity fade effect
- Blur transition for depth effect

### Dot Colors & Timing
```css
.dna-loader .dot:nth-child(1) { background: #22c55e; animation-delay: 0s; }
.dna-loader .dot:nth-child(2) { background: #38bdf8; animation-delay: 0.12s; }
.dna-loader .dot:nth-child(3) { background: #8b5cf6; animation-delay: 0.24s; }
.dna-loader .dot:nth-child(4) { background: #f59e0b; animation-delay: 0.36s; }
.dna-loader .dot:nth-child(5) { background: #ef4444; animation-delay: 0.48s; }
.dna-loader .dot:nth-child(6) { background: #22c55e; animation-delay: 0.6s; }
.dna-loader .dot:nth-child(7) { background: #38bdf8; animation-delay: 0.72s; }
```

**Color Scheme:**
- Green (#22c55e) - Health/positive
- Cyan (#38bdf8) - Medical/tech
- Purple (#8b5cf6) - AI/analysis
- Amber (#f59e0b) - Caution/processing
- Red (#ef4444) - Alert/urgent

### Overlay Fade-In: `fadeInOverlay`
```css
@keyframes fadeInOverlay {
    from {
        opacity: 0;
        backdrop-filter: blur(0px);
    }
    to {
        opacity: 1;
        backdrop-filter: blur(12px);
    }
}
```
**Duration:** 0.4s ease | Creates atmospheric backdrop

### Container Slide-Up: `slideInUp`
```css
@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
```
**Duration:** 0.6s ease with 0.2s delay | Smooth entrance

### Step Update Pulse: `slidePulse`
```css
@keyframes slidePulse {
    0%, 100% {
        box-shadow: 0 0 0 rgba(56, 189, 248, 0.4);
        opacity: 0.8;
    }
    50% {
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
        opacity: 1;
    }
}
```
**Duration:** 2s ease-in-out infinite | Draws attention to current step

### Progress Dot Bounce: `progressBounce`
```css
@keyframes progressBounce {
    0%, 100% {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1);
    }
    50% {
        background: #38bdf8;
        transform: scale(1.4);
    }
}
```
**Duration:** 1.4s ease infinite | Staggered bounce effect

### Text Fade-In: `fadeInText`
```css
@keyframes fadeInText {
    from { opacity: 0; }
    to { opacity: 1; }
}
```
**Duration:** 0.8s ease with variable delays

### Loading Text Gradient: `gradientShift`
```css
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
```
**For buttons:** 4s ease infinite

## JavaScript Event Handling

### Loading Animation Controller
```javascript
const LoadingAnimation = {
    stepIndex: 0,
    stepInterval: null,
    
    start() {
        // Disable button, show overlay
        // Start step interval (2.5s between updates)
    },
    
    stop() {
        // Clear interval
        // Enable button, hide overlay
    },
    
    reset() {
        // Full cleanup
    }
}
```

### File Upload Handler
```javascript
const FileUploadHandler = {
    handleFileSelect(event) {},
    displayFileInfo(file) {},
    displayPreview(file) {},
    init() {
        // Attach listeners for change, dragover, dragleave, drop
    }
}
```

## Performance Considerations

1. **CSS Animations**: Hardware accelerated (transform, opacity)
2. **Backdrop Filter**: Modern browsers only (60fps capable)
3. **Perspective**: 3D context for depth effects
4. **GPU Memory**: ~2-3MB for animation layers
5. **Battery Impact**: Low (GPU accelerated, minimal repaints)

## Browser Support

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| CSS Animation | ✅ | ✅ | ✅ | ✅ |
| Backdrop Filter | ✅ | ✅ (v103+) | ✅ (16.4+) | ✅ |
| CSS Filters | ✅ | ✅ | ✅ | ✅ |
| Cubic Bezier | ✅ | ✅ | ✅ | ✅ |
| Perspective | ✅ | ✅ | ✅ | ✅ |

## Mobile Optimization

**On Mobile Devices:**
- Loading text: 18px (desktop) → 16px (mobile)
- Dot size: 14px → 12px
- Gap between dots: 12px → 10px
- Step box min-width: 300px → 250px (tablet) → 200px (mobile)
- Animation unchanged for smooth performance

**Reduced Motion Support:**
```css
@media (prefers-reduced-motion: reduce) {
    .dna-loader .dot {
        animation: none;
        opacity: 0.7;
    }
    .loading-overlay {
        animation: none;
        opacity: 1;
    }
}
```
*(Can be added for accessibility)*

