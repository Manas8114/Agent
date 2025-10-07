/**
 * MetaMask Error Suppressor
 * Comprehensive solution to suppress MetaMask connection errors
 */

// Store original handlers
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;
const originalWindowError = window.onerror;

// Comprehensive MetaMask error detection
function isMetaMaskError(message) {
  if (typeof message === 'string') {
    return message.includes('MetaMask') || 
           message.includes('chrome-extension') ||
           message.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
           message.includes('Failed to connect to MetaMask') ||
           message.includes('inpage.js') ||
           message.includes('ethereum') ||
           message.includes('web3') ||
           message.includes('wallet') ||
           message.includes('connect') ||
           message.includes('nkbihfbeogaeaoehlefnkodbefgpgknn');
  }
  
  if (message && message.message) {
    return message.message.includes('MetaMask') || 
           message.message.includes('chrome-extension') ||
           message.message.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
           message.message.includes('Failed to connect to MetaMask') ||
           message.message.includes('inpage.js') ||
           message.message.includes('ethereum') ||
           message.message.includes('web3') ||
           message.message.includes('wallet') ||
           message.message.includes('connect');
  }
  
  if (message && message.stack) {
    return message.stack.includes('chrome-extension') ||
           message.stack.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
           message.stack.includes('inpage.js');
  }
  
  return false;
}

// Override console.error
console.error = function(...args) {
  const message = args.join(' ');
  if (!isMetaMaskError(message)) {
    originalConsoleError.apply(console, args);
  }
  // Silently suppress MetaMask errors
};

// Override console.warn
console.warn = function(...args) {
  const message = args.join(' ');
  if (!isMetaMaskError(message)) {
    originalConsoleWarn.apply(console, args);
  }
  // Silently suppress MetaMask warnings
};

// Override window.onerror
window.onerror = function(message, source, lineno, colno, error) {
  if (isMetaMaskError(message) || isMetaMaskError(error)) {
    // Silently suppress MetaMask errors
    return true; // Prevent default error handling
  }
  if (originalWindowError) {
    return originalWindowError(message, source, lineno, colno, error);
  }
  return false;
};

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
  if (isMetaMaskError(event.reason)) {
    // Silently suppress MetaMask promise rejections
    event.preventDefault();
    event.stopPropagation();
    return false;
  }
}, true);

// Additional error suppression
window.addEventListener('error', function(event) {
  if (isMetaMaskError(event.error) || isMetaMaskError(event.message)) {
    // Silently suppress MetaMask error events
    event.preventDefault();
    event.stopPropagation();
    return false;
  }
}, true);

// Suppress MetaMask injection attempts
const suppressMetaMaskInjection = () => {
  if (window.ethereum) {
    try {
      // Override ethereum.request to prevent errors
      const originalRequest = window.ethereum.request;
      if (originalRequest) {
        window.ethereum.request = function(...args) {
          try {
            return originalRequest.apply(this, args);
          } catch (error) {
            if (isMetaMaskError(error)) {
              // Return a rejected promise that won't cause unhandled rejection
              return Promise.reject(new Error('MetaMask connection suppressed'));
            }
            throw error;
          }
        };
      }
    } catch (e) {
      // Ignore any errors in the suppression setup
    }
  }
};

// Run suppression when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', suppressMetaMaskInjection);
} else {
  suppressMetaMaskInjection();
}

// Also run immediately
suppressMetaMaskInjection();

// Override React's error handling
const originalAddEventListener = window.addEventListener;
window.addEventListener = function(type, listener, options) {
  if (type === 'error' || type === 'unhandledrejection') {
    const wrappedListener = function(event) {
      if (isMetaMaskError(event.error || event.reason || event.message)) {
        event.preventDefault();
        event.stopPropagation();
        return false;
      }
      return listener.call(this, event);
    };
    return originalAddEventListener.call(this, type, wrappedListener, options);
  }
  return originalAddEventListener.call(this, type, listener, options);
};

console.log('🛡️ MetaMask error suppressor initialized');

export default {
  isMetaMaskError,
  suppressMetaMaskInjection
};
