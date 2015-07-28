(function(a,b,c){var d={html:false,photo:false,iframe:false,inline:false,transition:"elastic",speed:300,fadeOut:300,width:false,initialWidth:"600",innerWidth:false,maxWidth:false,height:false,initialHeight:"450",innerHeight:false,maxHeight:false,scalePhotos:true,scrolling:true,href:false,title:false,rel:false,opacity:0.9,preloading:true,className:false,overlayClose:true,escKey:true,arrowKey:true,top:false,bottom:false,left:false,right:false,fixed:false,data:undefined,closeButton:true,fastIframe:true,open:false,reposition:true,loop:true,slideshow:false,slideshowAuto:true,slideshowSpeed:2500,slideshowStart:"start slideshow",slideshowStop:"stop slideshow",photoRegex:/\.(gif|png|jp(e|g|eg)|bmp|ico|webp)((#|\?).*)?$/i,retinaImage:false,retinaUrl:false,retinaSuffix:'@2x.$1',current:"image {current} of {total}",previous:"previous",next:"next",close:"close",xhrError:"This content failed to load.",imgError:"This image failed to load.",returnFocus:true,trapFocus:true,onOpen:false,onLoad:false,onComplete:false,onCleanup:false,onClosed:false},e='colorbox',f='cbox',g=f+'Element',h=f+'_open',i=f+'_load',j=f+'_complete',k=f+'_cleanup',l=f+'_closed',m=f+'_purge',n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H=a('<a/>'),I,J,K,L,M,N,O,P,Q,R,S,T,U,V="div",W,X=0,Y={},Z;function ab(c,d,e){var g=b.createElement(c);if(d)g.id=f+d;if(e)g.style.cssText=e;return a(g);}function ac(){return c.innerHeight?c.innerHeight:a(c).height();}function ad(a){var b=v.length,c=(O+a)%b;return(c<0)?b+c:c;}function ae(a,b){return Math.round((/%/.test(a)?((b==='x'?w.width():ac())/100):1)*parseInt(a,10));}function af(a,b){return a.photo||a.photoRegex.test(b);}function ag(a,b){return a.retinaUrl&&c.devicePixelRatio>1?b.replace(a.photoRegex,a.retinaSuffix):b;}function ah(a){if('contains' in o[0]&&!o[0].contains(a.target)){a.stopPropagation();o.focus();}}function ai(){var b,c=a.data(N,e);if(c==null){I=a.extend({},d);if(console&&console.log)console.log('Error: cboxElement missing settings object');}else I=a.extend({},c);for(b in I)if(a.isFunction(I[b])&&b.slice(0,2)!=='on')I[b]=I[b].call(N);I.rel=I.rel||N.rel||a(N).data('rel')||'nofollow';I.href=I.href||a(N).attr('href');I.title=I.title||N.title;if(typeof I.href==="string")I.href=a.trim(I.href);}function aj(c,d){a(b).trigger(c);H.triggerHandler(c);if(a.isFunction(d))d.call(N);}var ak=(function(){var a,b=f+"Slideshow_",c="click."+f,d;function e(){clearTimeout(d);}function g(){if(I.loop||v[O+1]){e();d=setTimeout(U.next,I.slideshowSpeed);}}function h(){C.html(I.slideshowStop).unbind(c).one(c,l);H.bind(j,g).bind(i,e);o.removeClass(b+"off").addClass(b+"on");}function l(){e();H.unbind(j,g).unbind(i,e);C.html(I.slideshowStart).unbind(c).one(c,function(){U.next();h();});o.removeClass(b+"on").addClass(b+"off");}function m(){a=false;C.hide();e();H.unbind(j,g).unbind(i,e);o.removeClass(b+"off "+b+"on");}return function(){if(a){if(!I.slideshow){H.unbind(k,m);m();}}else if(I.slideshow&&v[1]){a=true;H.one(k,m);if(I.slideshowAuto)h();else l();C.show();}};}());function al(c){if(!S){N=c;ai();v=a(N);O=0;if(I.rel!=='nofollow'){v=a('.'+g).filter(function(){var b=a.data(this,e),c;if(b)c=a(this).data('rel')||b.rel||this.rel;return(c===I.rel);});O=v.index(N);if(O===-1){v=v.add(N);O=v.length-1;}}n.css({opacity:parseFloat(I.opacity),cursor:I.overlayClose?"pointer":"auto",visibility:'visible'}).show();if(W)o.add(n).removeClass(W);if(I.className)o.add(n).addClass(I.className);W=I.className;if(I.closeButton)F.html(I.close).appendTo(q);else F.appendTo('<div/>');if(!Q){Q=R=true;o.css({visibility:'hidden',display:'block'});x=ab(V,'LoadedContent','width:0; height:0; overflow:hidden');q.css({width:'',height:''}).append(x);J=r.height()+u.height()+q.outerHeight(true)-q.height();K=s.width()+t.width()+q.outerWidth(true)-q.width();L=x.outerHeight(true);M=x.outerWidth(true);I.w=ae(I.initialWidth,'x');I.h=ae(I.initialHeight,'y');x.css({width:'',height:I.h});U.position();aj(h,I.onOpen);G.add(A).hide();o.focus();if(I.trapFocus)if(b.addEventListener){b.addEventListener('focus',ah,true);H.one(l,function(){b.removeEventListener('focus',ah,true);});}if(I.returnFocus)H.one(l,function(){a(N).focus();});}ao();}}function am(){if(!o&&b.body){Z=false;w=a(c);o=ab(V).attr({id:e,'class':a.support.opacity===false?f+'IE':'',role:'dialog',tabindex:'-1'}).hide();n=ab(V,"Overlay").hide();z=a([ab(V,"LoadingOverlay")[0],ab(V,"LoadingGraphic")[0]]);p=ab(V,"Wrapper");q=ab(V,"Content").append(A=ab(V,"Title"),B=ab(V,"Current"),E=a('<button type="button"/>').attr({id:f+'Previous'}),D=a('<button type="button"/>').attr({id:f+'Next'}),C=ab('button',"Slideshow"),z);F=a('<button type="button"/>').attr({id:f+'Close'});p.append(ab(V).append(ab(V,"TopLeft"),r=ab(V,"TopCenter"),ab(V,"TopRight")),ab(V,false,'clear:left').append(s=ab(V,"MiddleLeft"),q,t=ab(V,"MiddleRight")),ab(V,false,'clear:left').append(ab(V,"BottomLeft"),u=ab(V,"BottomCenter"),ab(V,"BottomRight"))).find('div div').css({'float':'left'});y=ab(V,false,'position:absolute; width:9999px; visibility:hidden; display:none; max-width:none;');G=D.add(E).add(B).add(C);a(b.body).append(n,o.append(p,y));}}function an(){function c(a){if(!(a.which>1||a.shiftKey||a.altKey||a.metaKey||a.ctrlKey)){a.preventDefault();al(this);}}if(o){if(!Z){Z=true;D.click(function(){U.next();});E.click(function(){U.prev();});F.click(function(){U.close();});n.click(function(){if(I.overlayClose)U.close();});a(b).bind('keydown.'+f,function(a){var b=a.keyCode;if(Q&&I.escKey&&b===27){a.preventDefault();U.close();}if(Q&&I.arrowKey&&v[1]&&!a.altKey)if(b===37){a.preventDefault();E.click();}else if(b===39){a.preventDefault();D.click();}});if(a.isFunction(a.fn.on))a(b).on('click.'+f,'.'+g,c);else a('.'+g).live('click.'+f,c);}return true;}return false;}if(a.colorbox)return;a(am);U=a.fn[e]=a[e]=function(b,c){var f=this;b=b||{};am();if(an()){if(a.isFunction(f)){f=a('<a/>');b.open=true;}else if(!f[0])return f;if(c)b.onComplete=c;f.each(function(){a.data(this,e,a.extend({},a.data(this,e)||d,b));}).addClass(g);if((a.isFunction(b.open)&&b.open.call(f))||b.open)al(f[0]);}return f;};U.position=function(b,c){var d,e=0,g=0,h=o.offset(),i,j;w.unbind('resize.'+f);o.css({top:-9e4,left:-9e4});i=w.scrollTop();j=w.scrollLeft();if(I.fixed){h.top-=i;h.left-=j;o.css({position:'fixed'});}else{e=i;g=j;o.css({position:'absolute'});}if(I.right!==false)g+=Math.max(w.width()-I.w-M-K-ae(I.right,'x'),0);else if(I.left!==false)g+=ae(I.left,'x');else g+=Math.round(Math.max(w.width()-I.w-M-K,0)/2);if(I.bottom!==false)e+=Math.max(ac()-I.h-L-J-ae(I.bottom,'y'),0);else if(I.top!==false)e+=ae(I.top,'y');else e+=Math.round(Math.max(ac()-I.h-L-J,0)/2);o.css({top:h.top,left:h.left,visibility:'visible'});p[0].style.width=p[0].style.height="9999px";function k(){r[0].style.width=u[0].style.width=q[0].style.width=(parseInt(o[0].style.width,10)-K)+'px';q[0].style.height=s[0].style.height=t[0].style.height=(parseInt(o[0].style.height,10)-J)+'px';}d={width:I.w+M+K,height:I.h+L+J,top:e,left:g};if(b){var l=0;a.each(d,function(a){if(d[a]!==Y[a]){l=b;return;}});b=l;}Y=d;if(!b)o.css(d);o.dequeue().animate(d,{duration:b||0,complete:function(){k();R=false;p[0].style.width=(I.w+M+K)+"px";p[0].style.height=(I.h+L+J)+"px";if(I.reposition)setTimeout(function(){w.bind('resize.'+f,U.position);},1);if(c)c();},step:k});};U.resize=function(a){var b;if(Q){a=a||{};if(a.width)I.w=ae(a.width,'x')-M-K;if(a.innerWidth)I.w=ae(a.innerWidth,'x');x.css({width:I.w});if(a.height)I.h=ae(a.height,'y')-L-J;if(a.innerHeight)I.h=ae(a.innerHeight,'y');if(!a.innerHeight&&!a.height){b=x.scrollTop();x.css({height:"auto"});I.h=x.height();}x.css({height:I.h});if(b)x.scrollTop(b);U.position(I.transition==="none"?0:I.speed);}};U.prep=function(c){if(!Q)return;var d,g=I.transition==="none"?0:I.speed;x.empty().remove();x=ab(V,'LoadedContent').append(c);function h(){I.w=I.w||x.width();I.w=I.mw&&I.mw<I.w?I.mw:I.w;return I.w;}function i(){I.h=I.h||x.height();I.h=I.mh&&I.mh<I.h?I.mh:I.h;return I.h;}x.hide().appendTo(y.show()).css({width:h(),overflow:I.scrolling?'auto':'hidden'}).css({height:i()}).prependTo(q);y.hide();a(P).css({'float':'none'});d=function(){var c=v.length,d,h='frameBorder',i='allowTransparency',k;if(!Q)return;function l(){if(a.support.opacity===false)o[0].style.removeAttribute('filter');}k=function(){clearTimeout(T);z.hide();aj(j,I.onComplete);};A.html(I.title).add(x).show();if(c>1){if(typeof I.current==="string")B.html(I.current.replace('{current}',O+1).replace('{total}',c)).show();D[(I.loop||O<c-1)?"show":"hide"]().html(I.next);E[(I.loop||O)?"show":"hide"]().html(I.previous);ak();if(I.preloading)a.each([ad(-1),ad(1)],function(){var c,d,f=v[this],g=a.data(f,e);if(g&&g.href){c=g.href;if(a.isFunction(c))c=c.call(f);}else c=a(f).attr('href');if(c&&af(g,c)){c=ag(g,c);d=b.createElement('img');d.src=c;}});}else G.hide();if(I.iframe){d=ab('iframe')[0];if(h in d)d[h]=0;if(i in d)d[i]="true";if(!I.scrolling)d.scrolling="no";a(d).attr({src:I.href,name:new Date().getTime(),'class':f+'Iframe',allowFullScreen:true,webkitAllowFullScreen:true,mozallowfullscreen:true}).one('load',k).appendTo(x);H.one(m,function(){d.src="//about:blank";});if(I.fastIframe)a(d).trigger('load');}else k();if(I.transition==='fade')o.fadeTo(g,1,l);else l();};if(I.transition==='fade')o.fadeTo(g,0,function(){U.position(0,d);});else U.position(g,d);};function ao(){var d,e,g=U.prep,h,j=++X;R=true;P=false;N=v[O];ai();aj(m);aj(i,I.onLoad);I.h=I.height?ae(I.height,'y')-L-J:I.innerHeight&&ae(I.innerHeight,'y');I.w=I.width?ae(I.width,'x')-M-K:I.innerWidth&&ae(I.innerWidth,'x');I.mw=I.w;I.mh=I.h;if(I.maxWidth){I.mw=ae(I.maxWidth,'x')-M-K;I.mw=I.w&&I.w<I.mw?I.w:I.mw;}if(I.maxHeight){I.mh=ae(I.maxHeight,'y')-L-J;I.mh=I.h&&I.h<I.mh?I.h:I.mh;}d=I.href;T=setTimeout(function(){z.show();},100);if(I.inline){h=ab(V).hide().insertBefore(a(d)[0]);H.one(m,function(){h.replaceWith(x.children());});g(a(d));}else if(I.iframe)g(" ");else if(I.html)g(I.html);else if(af(I,d)){d=ag(I,d);P=b.createElement('img');a(P).addClass(f+'Photo').bind('error',function(){I.title=false;g(ab(V,'Error').html(I.imgError));}).one('load',function(){var b;if(j!==X)return;a.each(['alt','longdesc','aria-describedby'],function(b,c){var d=a(N).attr(c)||a(N).attr('data-'+c);if(d)P.setAttribute(c,d);});if(I.retinaImage&&c.devicePixelRatio>1){P.height=P.height/c.devicePixelRatio;P.width=P.width/c.devicePixelRatio;}if(I.scalePhotos){e=function(){P.height-=P.height*b;P.width-=P.width*b;};if(I.mw&&P.width>I.mw){b=(P.width-I.mw)/P.width;e();}if(I.mh&&P.height>I.mh){b=(P.height-I.mh)/P.height;e();}}if(I.h)P.style.marginTop=Math.max(I.mh-P.height,0)/2+'px';if(v[1]&&(I.loop||v[O+1])){P.style.cursor='pointer';P.onclick=function(){U.next();};}P.style.width=P.width+'px';P.style.height=P.height+'px';setTimeout(function(){g(P);},1);});setTimeout(function(){P.src=d;},1);}else if(d)y.load(d,I.data,function(b,c){if(j===X)g(c==='error'?ab(V,'Error').html(I.xhrError):a(this).contents());});}U.next=function(){if(!R&&v[1]&&(I.loop||v[O+1])){O=ad(1);al(v[O]);}};U.prev=function(){if(!R&&v[1]&&(I.loop||O)){O=ad(-1);al(v[O]);}};U.close=function(){if(Q&&!S){S=true;Q=false;aj(k,I.onCleanup);w.unbind('.'+f);n.fadeTo(I.fadeOut||0,0);o.stop().fadeTo(I.fadeOut||0,0,function(){o.add(n).css({'opacity':1,cursor:'auto'}).hide();aj(m);x.empty().remove();setTimeout(function(){S=false;aj(l,I.onClosed);},1);});}};U.remove=function(){if(!o)return;o.stop();a.colorbox.close();o.stop().remove();n.remove();S=false;o=null;a('.'+g).removeData(e).removeClass(g);a(b).unbind('click.'+f);};U.element=function(){return a(N);};U.settings=d;}(jQuery,document,window));