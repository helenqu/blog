/**
 * Bibliography/Citation System with BibTeX Support
 * 
 * This script handles:
 * 1. Parsing BibTeX data embedded in the page
 * 2. Numbering citations in order of first appearance
 * 3. Rendering bibliography entries
 * 4. Linking citations to bibliography entries
 */

(function() {
  'use strict';

  document.addEventListener('DOMContentLoaded', function() {
    initBibliography();
  });

  /**
   * Parse a BibTeX string into an object of entries
   */
  function parseBibtex(bibtexStr) {
    const entries = {};
    
    // Match BibTeX entries: @type{key, ... }
    const entryRegex = /@(\w+)\s*\{\s*([^,]+)\s*,([^@]*)\}/g;
    let match;
    
    while ((match = entryRegex.exec(bibtexStr)) !== null) {
      const type = match[1].toLowerCase();
      const key = match[2].trim();
      const fieldsStr = match[3];
      
      const entry = { type: type };
      
      // Parse fields: name = {value} or name = "value" or name = value
      const fieldRegex = /(\w+)\s*=\s*(?:\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|"([^"]*)"|(\d+))/g;
      let fieldMatch;
      
      while ((fieldMatch = fieldRegex.exec(fieldsStr)) !== null) {
        const fieldName = fieldMatch[1].toLowerCase();
        const fieldValue = (fieldMatch[2] || fieldMatch[3] || fieldMatch[4] || '').trim();
        entry[fieldName] = cleanLatex(fieldValue);
      }
      
      entries[key] = entry;
    }
    
    return entries;
  }

  /**
   * Clean LaTeX special characters
   */
  function cleanLatex(str) {
    return str
      .replace(/\\'/g, '́')  // Combining acute accent
      .replace(/\\'([aeiouAEIOU])/g, function(m, c) {
        const accents = {'a':'á','e':'é','i':'í','o':'ó','u':'ú','A':'Á','E':'É','I':'Í','O':'Ó','U':'Ú'};
        return accents[c] || c;
      })
      .replace(/\{|\}/g, '')
      .replace(/--/g, '–')
      .replace(/~/g, ' ')
      .replace(/\\\&/g, '&')
      .replace(/``/g, '"')
      .replace(/''/g, '"');
  }

  /**
   * Format an author string (Last, First and Last, First -> Last & Last)
   */
  function formatAuthors(authorStr) {
    if (!authorStr) return '';
    
    const authors = authorStr.split(/\s+and\s+/i);
    const formatted = authors.map(function(author) {
      author = author.trim();
      // If "Last, First" format, keep as is but clean up
      if (author.includes(',')) {
        const parts = author.split(',');
        return parts[0].trim() + ', ' + parts.slice(1).join(',').trim();
      }
      return author;
    });
    
    if (formatted.length === 1) return formatted[0];
    if (formatted.length === 2) return formatted[0] + ' and ' + formatted[1];
    return formatted.slice(0, -1).join(', ') + ', and ' + formatted[formatted.length - 1];
  }

  /**
   * Render a bibliography entry as HTML
   */
  function renderEntry(entry, key) {
    let html = '<span class="bib-authors">' + formatAuthors(entry.author) + '</span> ';
    html += '(' + (entry.year || '') + '). ';
    
    // Title
    if (entry.url) {
      html += '<span class="bib-title"><a href="' + entry.url + '" target="_blank" rel="noopener">' + entry.title + '</a></span>. ';
    } else {
      html += '<span class="bib-title">' + entry.title + '</span>. ';
    }
    
    // Type-specific formatting
    if (entry.type === 'article') {
      if (entry.journal) {
        html += '<em>' + entry.journal + '</em>';
      }
      if (entry.volume) html += ', ' + entry.volume;
      if (entry.number) html += '(' + entry.number + ')';
      if (entry.pages) html += ', ' + entry.pages;
      html += '. ';
    } else if (entry.type === 'inproceedings' || entry.type === 'incollection') {
      if (entry.booktitle) {
        html += 'In <em>' + entry.booktitle + '</em>';
      }
      if (entry.pages) html += ', pp. ' + entry.pages;
      html += '. ';
    } else if (entry.type === 'book') {
      if (entry.publisher) html += entry.publisher;
      if (entry.address) html += ', ' + entry.address;
      html += '. ';
    }
    
    // DOI
    if (entry.doi) {
      html += '<a href="https://doi.org/' + entry.doi + '" class="bib-doi" target="_blank" rel="noopener">doi:' + entry.doi + '</a>';
    }
    
    // Back-link
    html += ' <a href="#cite-' + key + '" class="bib-backlink" title="Back to citation">↩</a>';
    
    return html;
  }

  function initBibliography() {
    const citations = document.querySelectorAll('.citation[data-cite-key]');
    const bibSection = document.querySelector('.bibliography');
    
    if (!citations.length) return;

    // Parse BibTeX data if available
    let bibData = {};
    const bibtexScript = document.querySelector('script[type="text/bibtex"]');
    if (bibtexScript) {
      bibData = parseBibtex(bibtexScript.textContent);
    }

    // Track citation keys in order of first appearance
    const citationOrder = [];
    const citationMap = new Map();
    
    // First pass: determine order and assign numbers
    citations.forEach(function(cite) {
      const key = cite.dataset.citeKey;
      
      if (!citationMap.has(key)) {
        citationOrder.push(key);
        citationMap.set(key, citationOrder.length);
      }
    });

    // Second pass: update citation text and add IDs for back-links
    const citationCounts = new Map();
    
    citations.forEach(function(cite) {
      const key = cite.dataset.citeKey;
      const num = citationMap.get(key);
      
      const count = (citationCounts.get(key) || 0) + 1;
      citationCounts.set(key, count);
      
      cite.id = count === 1 ? 'cite-' + key : 'cite-' + key + '-' + count;
      cite.textContent = '[' + num + ']';
      cite.title = 'Jump to reference ' + num;
    });

    // Render bibliography
    if (bibSection && Object.keys(bibData).length > 0) {
      const bibList = bibSection.querySelector('.bib-list');
      
      if (bibList) {
        bibList.innerHTML = '';
        
        citationOrder.forEach(function(key) {
          const entry = bibData[key];
          
          if (entry) {
            const li = document.createElement('li');
            li.id = 'ref-' + key;
            li.className = 'bib-entry';
            li.dataset.bibKey = key;
            li.innerHTML = renderEntry(entry, key);
            bibList.appendChild(li);
          } else {
            console.warn('Bibliography entry not found for key: ' + key);
          }
        });
      }
    }

    // Add smooth scrolling for citation links
    document.querySelectorAll('.citation, .bib-backlink').forEach(function(link) {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').slice(1);
        const target = document.getElementById(targetId);
        
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'center' });
          
          target.classList.add('highlight');
          setTimeout(function() {
            target.classList.remove('highlight');
          }, 2000);
        }
      });
    });
  }
})();
