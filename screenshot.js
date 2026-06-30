const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 800 });
  await page.goto('file:///home/dingkwang/sci/llm-rl-course/lecture_L3.html');
  
  // Wait a bit for fonts to load
  await new Promise(r => setTimeout(r, 1000));
  
  // Find the exact element to screenshot
  const element = await page.$('.insight.reveal');
  
  if (element) {
    // There are multiple .insight.reveal elements, we want the one with "大白话拆解"
    const elements = await page.$$('.insight.reveal');
    for (let el of elements) {
      const text = await page.evaluate(e => e.textContent, el);
      if (text.includes('大白话拆解')) {
        await el.screenshot({ path: 'screenshot_code.png' });
        break;
      }
    }
  } else {
    await page.screenshot({ path: 'screenshot_code.png' });
  }
  
  await browser.close();
})();
