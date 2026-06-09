// ===== PARTICLE BACKGROUND =====
const canvas = document.getElementById('particles-canvas');
const ctx = canvas.getContext('2d');
let particles = [];

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

class Particle {
  constructor() { this.reset(); }
  reset() {
    this.x = Math.random() * canvas.width;
    this.y = Math.random() * canvas.height;
    this.size = Math.random() * 2 + 0.5;
    this.speedX = (Math.random() - 0.5) * 0.3;
    this.speedY = (Math.random() - 0.5) * 0.3;
    this.opacity = Math.random() * 0.4 + 0.1;
  }
  update() {
    this.x += this.speedX;
    this.y += this.speedY;
    if (this.x < 0 || this.x > canvas.width || this.y < 0 || this.y > canvas.height) this.reset();
  }
  draw() {
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(74,125,255,${this.opacity})`;
    ctx.fill();
  }
}

for (let i = 0; i < 80; i++) particles.push(new Particle());

function animateParticles() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach(p => { p.update(); p.draw(); });
  // draw connections
  for (let a = 0; a < particles.length; a++) {
    for (let b = a + 1; b < particles.length; b++) {
      const dx = particles[a].x - particles[b].x;
      const dy = particles[a].y - particles[b].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 120) {
        ctx.beginPath();
        ctx.strokeStyle = `rgba(74,125,255,${0.06 * (1 - dist / 120)})`;
        ctx.lineWidth = 0.5;
        ctx.moveTo(particles[a].x, particles[a].y);
        ctx.lineTo(particles[b].x, particles[b].y);
        ctx.stroke();
      }
    }
  }
  requestAnimationFrame(animateParticles);
}
animateParticles();

// ===== NAVBAR SCROLL =====
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
});

// ===== SCROLL ANIMATIONS =====
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.15 });

document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

// ===== COMPONENT DATA =====
const componentData = {
  electromagnet: {
    title: 'Electromagnet',
    subtitle: 'Custom-Wound Solenoid Coils',
    image: 'electromagnet.png',
    description: 'Each electromagnet is a custom-wound solenoid coil designed to generate precise magnetic field gradients. The coils use enameled copper wire wound around a ferrite core with a mounting flange. Eight electromagnets are arranged in a 3D configuration to provide full spatial control of the magnetic field vector.',
    specs: [
      { label: 'Wire Gauge', value: '22 AWG' },
      { label: 'Core Material', value: 'Ferrite' },
      { label: 'Turns', value: '~200' },
      { label: 'Max Current', value: '5A' },
      { label: 'Configuration', value: '8 Coils' },
      { label: 'Field Strength', value: '~50 mT' },
    ]
  },
  submodule: {
    title: 'Submodule',
    subtitle: '3-Axis Electromagnet Junction',
    image: 'submodule.png',
    description: 'The submodule is the structural backbone of the system — a 3D-printed junction that holds an electromagnet along each of the three orthogonal axes (X, Y, Z). Four submodules are combined to create the full octupole arrangement, enabling complete 3-axis control over the magnetic field gradient within the workspace.',
    specs: [
      { label: 'Material', value: '3D Printed PLA' },
      { label: 'Axes', value: '3 (X, Y, Z)' },
      { label: 'Units Used', value: '4 Total' },
      { label: 'Design', value: 'CAD (SolidWorks)' },
      { label: 'Assembly', value: 'Press-Fit' },
      { label: 'Configuration', value: 'Octupole' },
    ]
  },
  teensy: {
    title: 'Teensy 4.1',
    subtitle: 'High-Speed Microcontroller',
    image: 'teensy.jpg',
    description: 'The Teensy 4.1, powered by the NXP i.MX RT1062 ARM Cortex-M7 processor at 600 MHz, serves as the central controller. It runs the real-time MPC (Model Predictive Control) algorithm, reads sensor feedback, and outputs PWM signals to all 8 motor drivers simultaneously with microsecond-level timing precision.',
    specs: [
      { label: 'Processor', value: 'ARM Cortex-M7' },
      { label: 'Clock Speed', value: '600 MHz' },
      { label: 'PWM Channels', value: '8 Used' },
      { label: 'ADC Resolution', value: '12-bit' },
      { label: 'Protocol', value: 'USB Serial' },
      { label: 'Control Loop', value: '1 kHz' },
    ]
  },
  vascular: {
    title: 'Vascular Phantom',
    subtitle: 'Biomimetic Flow Channel',
    image: 'vasc.png',
    description: 'The vascular phantom is a transparent 3D-printed model that simulates a branching blood vessel network. It provides a realistic testing environment for the nanoparticle steering system, featuring bifurcations of varying diameters to mimic arterial branching. Fluid is pumped through to simulate blood flow conditions.',
    specs: [
      { label: 'Material', value: 'Clear Resin' },
      { label: 'Channel Dia.', value: '1–5 mm' },
      { label: 'Branches', value: 'Multi-furcation' },
      { label: 'Fabrication', value: 'SLA 3D Print' },
      { label: 'Flow Medium', value: 'Water + Glycerol' },
      { label: 'Transparency', value: 'Optically Clear' },
    ]
  },
  motordriver: {
    title: 'Motor Driver',
    subtitle: 'IBT-2 H-Bridge Power Module',
    image: 'motordriver.jpg',
    description: 'The IBT-2 motor driver modules control current flow through each electromagnet. Using the BTS7960 dual H-bridge IC, each driver can deliver up to 43A and supports bidirectional current control via PWM. Eight drivers are used — one per electromagnet — enabling independent and rapid polarity switching for dynamic field control.',
    specs: [
      { label: 'IC', value: 'BTS7960' },
      { label: 'Max Current', value: '43A Peak' },
      { label: 'Voltage', value: '6–27V' },
      { label: 'Control', value: 'PWM Input' },
      { label: 'Drivers Used', value: '8 Total' },
      { label: 'Direction', value: 'Bidirectional' },
    ]
  }
};

// ===== MODAL LOGIC =====
const overlay = document.getElementById('detail-overlay');
const modal = document.getElementById('detail-modal');

function openDetail(key) {
  const d = componentData[key];
  if (!d) return;
  let specsHTML = d.specs.map(s =>
    `<div class="spec-item"><div class="spec-label">${s.label}</div><div class="spec-value">${s.value}</div></div>`
  ).join('');

  modal.innerHTML = `
    <button class="modal-close" onclick="closeDetail()">&times;</button>
    <div class="modal-header">
      <img src="${d.image}" alt="${d.title}" class="modal-image">
      <div class="modal-info">
        <h2>${d.title}</h2>
        <div class="subtitle">${d.subtitle}</div>
        <p>${d.description}</p>
      </div>
    </div>
    <div class="specs-grid">${specsHTML}</div>
  `;
  overlay.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeDetail() {
  overlay.classList.remove('active');
  document.body.style.overflow = '';
}

overlay.addEventListener('click', (e) => {
  if (e.target === overlay) closeDetail();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeDetail();
});

// ===== SMOOTH NAV =====
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', () => {
    const key = link.dataset.section;
    if (key === 'home') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (key) {
      openDetail(key);
    }
  });
});

document.querySelector('.nav-logo').addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
