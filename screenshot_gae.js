const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 1600 });
  await page.goto('file:///home/dingkwang/sci/llm-rl-course/lecture_L3.html', {waitUntil: 'networkidle0'});
  
  await new Promise(r => setTimeout(r, 2000));
  
  await page.evaluate(() => {
    document.querySelectorAll('.reveal').forEach(el => {
      el.style.opacity = '1';
      el.style.transform = 'none';
      el.classList.remove('reveal');
    });
  });

  const elements = await page.$$('.code');
  for (let el of elements) {
    const text = await page.evaluate(e => e.textContent, el);
    if (text.includes('nextnonterminal')) {
      await el.screenshot({ path: 'screenshot_nextnonterminal.png' });
      break;
    }
  }
  
  await browser.close();
})();
