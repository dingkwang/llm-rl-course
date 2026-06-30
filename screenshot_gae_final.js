const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 1800 });
  await page.goto('file:///home/dingkwang/sci/llm-rl-course/lecture_L3.html#gae', {waitUntil: 'networkidle0'});
  
  await new Promise(r => setTimeout(r, 2000));
  
  await page.evaluate(() => {
    document.querySelectorAll('.reveal').forEach(el => {
      el.style.opacity = '1';
      el.style.transform = 'none';
      el.classList.remove('reveal');
    });
  });

  const section = await page.$('#gae');
  if (section) {
    await section.screenshot({ path: 'screenshot_gae_section.png' });
  }
  
  await browser.close();
})();
