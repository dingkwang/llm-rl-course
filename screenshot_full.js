const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 1600 });
  await page.goto('file:///home/dingkwang/sci/llm-rl-course/lecture_L3.html', {waitUntil: 'networkidle0'});
  
  // Wait a bit for fonts to load and animations to finish
  await new Promise(r => setTimeout(r, 2000));
  
  // Remove reveal class from elements to make sure they are visible
  await page.evaluate(() => {
    document.querySelectorAll('.reveal').forEach(el => {
      el.style.opacity = '1';
      el.style.transform = 'none';
      el.classList.remove('reveal'); // Prevent CSS animations from hiding it
    });
  });

  const elements = await page.$$('.insight');
  for (let el of elements) {
    const text = await page.evaluate(e => e.textContent, el);
    if (text.includes('大白话拆解')) {
      await el.screenshot({ path: 'screenshot_code2.png' });
      break;
    }
  }
  
  await browser.close();
})();
