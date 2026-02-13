/**
 * Image Lightbox
 * 
 * Provides a full-screen popup view for images.
 * - Click on any image to open lightbox
 * - Close via X button, clicking outside, or pressing Escape
 */

(function() {
  'use strict';

  let overlay = null;
  let lightboxImg = null;
  let lightboxCaption = null;

  function createLightbox() {
    // Create overlay
    overlay = document.createElement('div');
    overlay.className = 'lightbox-overlay';
    
    // Create close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'lightbox-close';
    closeBtn.setAttribute('aria-label', 'Close lightbox');
    
    // Create content container
    const content = document.createElement('div');
    content.className = 'lightbox-content';
    
    // Create image element
    lightboxImg = document.createElement('img');
    lightboxImg.alt = '';
    
    // Create caption element
    lightboxCaption = document.createElement('div');
    lightboxCaption.className = 'lightbox-caption';
    
    content.appendChild(lightboxImg);
    content.appendChild(lightboxCaption);
    overlay.appendChild(closeBtn);
    overlay.appendChild(content);
    document.body.appendChild(overlay);

    // Event listeners for closing
    closeBtn.addEventListener('click', closeLightbox);
    
    overlay.addEventListener('click', function(e) {
      // Close if clicking on overlay background (not on the image or caption)
      if (e.target === overlay) {
        closeLightbox();
      }
    });

    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && overlay.classList.contains('active')) {
        closeLightbox();
      }
    });
  }

  function getCaption(img) {
    // Try to find caption from figcaption (if image is inside a <figure>)
    const figure = img.closest('figure');
    if (figure) {
      const figcaption = figure.querySelector('figcaption');
      if (figcaption) {
        return figcaption.textContent.trim();
      }
    }
    
    // Fall back to alt text
    if (img.alt) {
      return img.alt;
    }
    
    // Fall back to title attribute
    if (img.title) {
      return img.title;
    }
    
    return '';
  }

  function openLightbox(src, caption) {
    if (!overlay) createLightbox();
    
    lightboxImg.src = src;
    lightboxCaption.textContent = caption || '';
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';
    
    // Show overlay
    overlay.classList.add('active');
  }

  function closeLightbox() {
    if (!overlay) return;
    
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  }

  function init() {
    // Select all images in content areas (excluding site icons, etc.)
    const images = document.querySelectorAll('article img, main img, .content img');
    
    images.forEach(function(img) {
      // Skip small images (likely icons) and already processed
      if (img.classList.contains('site-icon') || img.dataset.lightbox === 'disabled') {
        return;
      }
      
      img.addEventListener('click', function(e) {
        e.preventDefault();
        const caption = getCaption(this);
        openLightbox(this.src, caption);
      });
    });
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

