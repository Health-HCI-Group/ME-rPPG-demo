## Web Browser Inference Demo
Our fully optimized web deployment is available here, which implements multi-threading acceleration. 
[https://rppgdemo.kegang.wang/](https://rppgdemo.kegang.wang/) 

## Deployment Guidelines 
Ensure you possess a registered domain name with corresponding SSL certificates, as certain devices only permit camera access through HTTPS-enabled webpages.  
Configure Nginx with the following HTTP headers:  
`add_header Cross-Origin-Opener-Policy same-origin always;`  
`add_header Cross-Origin-Embedder-Policy require-corp always;`  
Failure to implement these directives will disable SIMD optimization, restricting the model to single-core CPU execution and significantly degrading inference performance.