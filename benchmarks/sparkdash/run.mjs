/** Owner-authorized stock SparkDash campaign; no changes to upstream functions. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
assert(process.env.SPARKDASH_EXCLUSIVE === '1', 'Close public ingress and declare SPARKDASH_EXCLUSIVE=1');
assert(process.argv[2], 'Pass a fresh output directory');
const output = path.resolve(process.argv[2]);
assert(!fs.existsSync(path.join(output, 'CAMPAIGN.json')), 'Never overwrite an existing run');
fs.mkdirSync(output, {recursive:true, mode:0o700});
for (const [key, file] of Object.entries({BENCH_HISTORY_PATH:'decode-history.json',
  BENCH_ACTIVE_PATH:'decode-active.json',PREFILL_BENCH_HISTORY_PATH:'prefill-history.json',
  PREFILL_BENCH_ACTIVE_PATH:'prefill-active.json'})) process.env[key]=path.join(output,file);

const provenance=JSON.parse(fs.readFileSync(path.join(here,'PROVENANCE.json'),'utf8'));
for(const [name, digest] of Object.entries(provenance.files)){
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(here,'vendor',name))).digest('hex'),digest,name);
}
const expected='a79dd73d621397c5fd159b797850ef5d4ec8246b10bc89b96e26ffe2ee38dc2c';
if(process.env.SPARKDASH_NATIVE_PID){
  const pid=process.env.SPARKDASH_NATIVE_PID;assert(/^[1-9][0-9]*$/.test(pid));
  const actual=crypto.createHash('sha256').update(fs.readFileSync('/proc/'+pid+'/exe')).digest('hex');
  assert.equal(actual,process.env.SPARKDASH_BINARY_SHA256 || expected,'Selected native identity');
}
const port=Number(process.env.SPARKDASH_PORT || 8003);
assert(Number.isInteger(port)&&port>0&&port<65536);
const baseUrl='http://127.0.0.1:'+port;
const health=await (await fetch(baseUrl+'/health')).json();
assert.equal(health.live.connections,0,'Native backend must be idle');
assert.equal(health.live.waiting,0);
const {decodeBenchManager}=await import('./vendor/server/collectors/DecodeBench.js');
const {prefillBenchManager}=await import('./vendor/server/collectors/PrefillBench.js');
const {LLM_STREAM_AGENT}=await import('./vendor/server/collectors/LlmStreaming.js');
const campaign={version:'1.2.5',measuredRuntimeAlias:'1.2.4k2a',startedAt:Date.now(),
  upstream:provenance,model:process.env.SPARKDASH_MODEL || 'deepseek-v4.1-flash',hardware:'3 x DGX Spark / TP3',
  referenceHeadBinarySha256:expected,thinking:false,temperature:0,top_p:1,
  maxTokens:400,concurrencies:[1,2,3,4,5,6,8,10,12],contextSizes:[1024,2048,4096,8192],
  nativeRestarted:false,publicApiHeldClosed:true,status:'running',jobs:{}};
const save=()=>fs.writeFileSync(path.join(output,'CAMPAIGN.json'),JSON.stringify(campaign,null,2));
save();
let current=null;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function waitJob(manager,job,name){
  current={manager,job};let previous='';
  while(true){
    const value=manager.getJob(job.benchId);
    const progress=JSON.stringify(value.progress);
    if(progress!==previous){console.log(JSON.stringify({event:'progress',name,status:value.status,...value.progress}));previous=progress;}
    if(value.status!=='running'){
      fs.writeFileSync(path.join(output,name+'.json'),JSON.stringify(value,null,2));
      campaign.jobs[name]=value;save();current=null;
      assert.equal(value.status,'completed',name+': '+value.error);
      assert(!value.error,name+': '+value.error);
      return value;
    }
    await sleep(1000);
  }
}
try{
  for(const promptType of ['prose','code']){
    const job=decodeBenchManager.start({sparkId:'tensorfold-125',lanIp:'127.0.0.1',port,
      modelId:campaign.model,concurrencies:campaign.concurrencies,maxTokens:400,promptType,
      debug:true,owner:{id:'operator-exclusive-benchmark',via:'public-wrapper'}});
    const value=await waitJob(decodeBenchManager,job,promptType);
    assert.equal(value.results.length,9);
    for(const row of value.results){
      assert(!row.error,`${promptType} C${row.concurrency}: ${row.error}`);
      assert.equal(row.streamsOk,row.concurrency);
      assert.equal(row.streamsFailed,0);
      assert.equal(row.totalCompletionTokens,400*row.concurrency);
      for(const stream of row.streams){
        assert.equal(stream.reasoningChunks,0,'Requested thinking OFF must actually be OFF');
        assert(stream.ttftContentMs!==null,'Actual answer tokens must be observed');
        assert.equal(stream.completionTokens,400);
      }
    }
    console.log(JSON.stringify({event:'decode_complete',type:promptType,
      rows:value.results.map(r=>({C:r.concurrency,aggregate:r.aggregateDecodeTps,stream:r.meanDecodeTps,ttft:r.meanTtftMs}))}));
  }
  const job=prefillBenchManager.start({sparkId:'tensorfold-125',lanIp:'127.0.0.1',port,
    modelId:campaign.model,contextSizes:campaign.contextSizes,contextLength:401408,
    owner:{id:'operator-exclusive-benchmark',via:'public-wrapper'}});
  const value=await waitJob(prefillBenchManager,job,'prefill');
  assert.equal(value.results.length,4);
  for(const row of value.results){assert(!row.error,row.error);assert.equal(row.samples,row.samplesRequested);assert(row.prefillTps>0);}
  campaign.status='completed';campaign.finishedAt=Date.now();save();
  console.log(JSON.stringify({event:'complete',durationMs:campaign.finishedAt-campaign.startedAt,output}));
}catch(error){
  campaign.status='failed';campaign.error=String(error);campaign.finishedAt=Date.now();save();
  if(current)current.manager.cancel('tensorfold-125',current.job.benchId);
  console.error(error);process.exitCode=1;
}finally{await LLM_STREAM_AGENT.close();}
