// VisiSign App UI Controller & Event Handler
document.addEventListener('DOMContentLoaded', () => {
    console.log('VisiSign App UI Engine Initialized');

    // Setup Aksi Tombol CTA Utama
    setupCTAButtons();

    // Setup Transisi Alur Gerak Halus (Smooth Scroll)
    setupSmoothScroll();

    // Setup Respons Menu Mobile
    setupMobileMenu();

    // Setup Efek Animasi Kartu Interaktif
    setupCardParallaxEffects();
});

// 1. MANAJEMEN TOMBOL CTA HERO SECTION
const setupCTAButtons = () => {
    const learnMoreBtn = document.querySelector('a[href="#panduan"]');
    if (learnMoreBtn) {
        learnMoreBtn.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('panduan')?.scrollIntoView({ behavior: 'smooth' });
        });
    }
};

// 2. LOGIKA SMOOTH SCROLL UNTUK NAVIGASI ANCHOR
const setupSmoothScroll = () => {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#' && href !== '#0') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
};

// 3. RESPONS MENU RESPONSIVE (MOBILE MENU TOGGLE)
const setupMobileMenu = () => {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }
};

// 4. ANIMATION OBSERVER UNTUK MERINGANKAN BEBAN BROWSING (ON-SCROLL EFFECTS)
document.addEventListener('DOMContentLoaded', () => {
    const observerOptions = {
        root: null,
        rootMargin: '0px 0px -10% 0px',
        threshold: 0.1
    };

    const observerCallback = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                
                // Jika elemen tersebut memiliki animasi counter data statistik
                if (entry.target.hasAttribute('id') && entry.target.id.startsWith('stat-')) {
                    const counterId = entry.target.id;
                    const endValue = parseInt(entry.target.getAttribute('data-end-value'));
                    animateCounter(counterId, endValue);
                }
                observer.unobserve(entry.target);
            }
        });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);
    document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));
});

// 5. ENGINES INDIKATOR GRAFIK STATISTIK (COUNTER ANIMATION)
const animateCounter = (id, endValue) => {
    const counterElement = document.getElementById(id);
    if (!counterElement) return;

    const duration = 1500; 
    const startValue = 0;
    const stepTime = Math.abs(Math.floor(duration / endValue));
    const increment = endValue > 3000 ? 50 : 1; 

    let currentValue = startValue;
    counterElement.innerText = currentValue;

    const interval = setInterval(() => {
        currentValue += increment;

        if (currentValue >= endValue) {
            currentValue = endValue;
            clearInterval(interval);
        }
        counterElement.innerText = currentValue.toLocaleString() + (endValue > 1 ? '+' : '');
    }, stepTime);
};

// 6. EFEK GLOWING GRADIENT PADA KARTU SEATAS KURSOR BERGERAK (INTERACTIVE CARDS)
const setupCardParallaxEffects = () => {
    document.querySelectorAll('.bg-white.rounded-3xl').forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            card.style.setProperty(
                'background',
                `linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(99,102,241,0.03) ${x}%, rgba(255,255,255,1) 100%)`
            );
        });
        
        // Kembalikan ke warna background dasar ketika kursor keluar dari area kartu
        card.addEventListener('mouseleave', () => {
            card.style.background = '#ffffff';
        });
    });
};