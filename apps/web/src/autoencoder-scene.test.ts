import { describe, expect, it } from "vitest";
import { AE_DURATION, AE_STAGES, AE_NUMBER_Y, autoencoderFrame, autoencoderFlow, autoencoderNodeBounds, drawAutoencoder, type ScenePalette } from "./autoencoder-scene";

describe("autoencoder architecture and timeline", () => {
  it("preserves the supplied convolution, pooling and decoder dimensions", () => {
    const shapes = AE_STAGES.map(s => [...s.shape]);
    expect(shapes[0]).toEqual([1,32,64]);
    expect(shapes[1]).toEqual([16,32,64]);
    expect(shapes[2]).toEqual([16,16,32]);
    expect(shapes[3]).toEqual([32,16,32]);
    expect(shapes[4]).toEqual([32,8,16]);
    expect(shapes[5]).toEqual([16,16,32]);
    expect(shapes[6]).toEqual(shapes[0]);
    expect(shapes[4].reduce<number>((a,b)=>a*b,1)).toBe(4096);
  });
  it("allows exact stage boundaries and holds the completed last frame", () => {
    AE_STAGES.forEach((stage,i)=>expect(autoencoderFrame(stage.start)).toMatchObject({index:i,progress:0}));
    expect(autoencoderFrame(AE_DURATION)).toEqual({time:24,index:6,progress:1});
    expect(autoencoderFrame(999)).toEqual(autoencoderFrame(24));
    expect(autoencoderFrame(-1)).toEqual(autoencoderFrame(0));
    expect(autoencoderFrame(NaN)).toEqual(autoencoderFrame(0));
  });
  it("hands off continuously at every boundary and settles on each chosen layer", () => {
    AE_STAGES.forEach((stage,i)=>{
      const at=autoencoderFlow(stage.start);
      expect(at.position).toBe(i);
      expect(at.weights[i]).toBe(1);
      if(i>0){
        expect(Math.abs(autoencoderFlow(stage.start-.0001).position-at.position)).toBeLessThan(.0001);
        expect(Math.abs(autoencoderFlow(stage.start+.0001).position-at.position)).toBeLessThan(.0001);
      }
    });
  });
  it("keeps a bounded, monotonic cursor and conserves the layer highlight", () => {
    let previous=0;
    for(let t=0;t<=AE_DURATION;t+=.01){
      const flow=autoencoderFlow(t);
      expect(flow.position).toBeGreaterThanOrEqual(previous-1e-9);
      expect(flow.position-previous).toBeLessThan(.025);
      expect(flow.weights.reduce((a,b)=>a+b,0)).toBeCloseTo(1);
      previous=flow.position;
    }
  });
  it("reserves a clear number row above every projected tensor", () => {
    AE_STAGES.forEach((_,i)=>expect(autoencoderNodeBounds(i).top-(AE_NUMBER_Y+4)).toBeGreaterThan(16));
  });
  it("renders identical drawing commands after seeking away and back", () => {
    const commands: unknown[] = [];
    const ctx = new Proxy({}, {set:(_target,key,value)=>{commands.push([key,value]);return true;},get:(_target,key)=>(...args:unknown[])=>commands.push([key,...args])}) as CanvasRenderingContext2D;
    const palette:ScenePalette={background:"#fff",text:"#000",muted:"#555",grid:"#ccc",accent:"#06f",input:"#ff0",encoder:"#fa0",latent:"#90f",decoder:"#06f"};
    const capture=(t:number)=>{commands.length=0;drawAutoencoder(ctx,1200,350,t,palette);return JSON.stringify(commands);};
    const first=capture(7.5);capture(21);capture(2);
    expect(capture(7.5)).toBe(first);
  });
  it("removes moving kernel and arrow particles in reduced-motion mode", () => {
    const commands: unknown[]=[];
    const ctx=new Proxy({}, {set:()=>true,get:(_target,key)=>(...args:unknown[])=>commands.push([key,...args])}) as CanvasRenderingContext2D;
    const palette:ScenePalette={background:"#fff",text:"#000",muted:"#555",grid:"#ccc",accent:"#06f",input:"#ff0",encoder:"#fa0",latent:"#90f",decoder:"#06f"};
    drawAutoencoder(ctx,420,330,3,palette,true);
    expect(commands.filter(c=>Array.isArray(c)&&c[0]==="fillRect")).toHaveLength(1);
  });
});
