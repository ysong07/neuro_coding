% 
% for i = 1:length(obj.delay)
%     delay(:,:,i) = obj.delay{i};
% end
% for i = 1:length(obj.energy)
%     energy(:,:,i) = obj.energy{i};
% end
% for i = 1:length(obj.mask)
%     mask(:,:,i) = obj.mask{i};
% end
% delay = reshape(delay,18*20,[]);
% energy = reshape(energy,18*20,[]);
% mask = reshape(mask,18*20,[]);
% save('delay_40','delay','energy','mask')


%% clustering result in one figure
clear all 
clc
a= load('data_40_Dirichlet_correlation_isomap_label');
labels = a.labels;
%%
a = load('all_correlation_40');
a = load('data_40_delay');
energy = a.energy;
delay = a.delay;
mask = a.mask;
a = load('obj_data_40');
obj = a.obj;

% a = load('features_and_delay_map_new');
% energy = a.energy;
% delay = a.delay;
% mask = a.mask;
% addpath('/Users/songyilin/Documents/bugrathesis/ms_thesis_results_code/data');
% a = load('objnew');
% obj = a.obj;
delay = reshape(delay,18,20,[]);
energy = reshape(energy,18,20,[]);
mask = reshape(mask,18,20,[]);

labels = labels+1;
for i = unique(labels)'
    a = find(labels ==i);
    cluster{i} = a(1:1:length(a));
end
%  
for i = unique(labels)'
    
%     time = cell2mat(obj.end(A(cluster{i}))) - cell2mat(obj.start(A(cluster{i})));
%     mean_time = floor(mean(time));
%     var_time = floor(var(time).^0.5);
%     num_time = length(time);
%     file_M = num2str(i);
%     file_name = [' mean ', num2str(mean_time),' var ',num2str(var_time),' num ', num2str(num_time)]; 
%  
%     
%     time = cell2mat(obj.end(A(cluster{i}))) - cell2mat(obj.start(A(cluster{i})));
%     mean_time = floor(mean(time));
        

    
    if (length(cluster{i})<50)
        figure(1)
         set (gcf,'Position',[200,100,600,1500], 'color','w')     
        for j = 1:length(cluster{i})     
        
        
    % delay map    
        n = floor(length(cluster{i}).^0.5) +1;
        signal_length = size(obj.model{(cluster{i}(j))},3);    
        subplot(n,n,j)
    % rescaling    
        min_val = 0;
        val = reshape(delay(:,:,(cluster{i}(j))),1,[]);
        val = sort(val,'descend');
        abcd = val(find(val<100));
        max_val = abcd(1);
        
        
        colormap([jet(256);gray(2)]);
        h(j) = imagesc(delay(:,:,(cluster{i}(j))));
          for abc = 1:18
                for efg = 1:20
                    if (mask(abc,efg,(cluster{i}(j)))==0)
                        C(abc,efg,j) = 258;
                    else 
                        m = 255; 
                        C(abc,efg,j) = min(m,floor(m*(delay(abc,efg,(cluster{i}(j)))-min_val)/(max_val-min_val))+1); 

                    end
                end
          end
        if sum(sum(mask(:,:,(cluster{i}(j))))) ~=360    
        set(h(j),'CData',C(:,:,j))
        caxis([min(min(C(:,:,j))),max(max(C(:,:,j)))]) 

        else
        set(h(j),'CData',C(:,:,j))
        caxis([min(min(C(:,:,j))),max(max(C(:,:,j)))+2])   
        end

    %     shading interp
        axis off  
        title(num2str((cluster{i}(j))));

        end 
     subplot(n,n,j+1)
     
     %rescaling
         val_cluster = [];
    for j = 1:length(cluster{i})   
        val = reshape(delay(:,:,(cluster{i}(j))),1,[]);
        val_cluster = cat(2,val,val_cluster);
    end
        val_cluster = sort(val_cluster,'descend');
         min_val = 0;
         abcd = val_cluster(find(val_cluster<100));
         max_val = abcd(floor(length(abcd)*0.05)+1);
%         max_val = val_cluster(floor(0.05*length(val_cluster)));
%         min_val = val_cluster(floor(0.95*length(val_cluster)));
%         val_cluster = sort(unique(val_cluster),'descend');
%         max_val = val_cluster(floor(0.40*length(val_cluster)));
%         min_val = val_cluster(floor(0.95*length(val_cluster)));
     
     
%      h1 = imagesc(sum(delay(:,:,A(cluster{i})),3)./size(delay(:,:,A(cluster{i})),3),[min_val max_val]);
     label_vector = floor((0:1/5*(max_val-min_val): (max_val-min_val)));
     h2 = colorbar('XTickLabel',{num2str(label_vector(1)),num2str(label_vector(2)),...
     num2str(label_vector(3)),num2str(label_vector(4)),num2str(label_vector(5)),num2str(label_vector(6))},'location','southoutside');
    
     axis off
     set(h2,'position',[.04 .05 .92 .05])
     
     
     
    % output
     file_name = num2str(i);
          mkdir('./test24/');
   
    file_N = ['./test24/',file_name];
    hgexport(figure (1),file_N)
    close figure 1
    
    else
        
        clus = cluster{i};
        for k = 1:50:length(clus)
            
            figure(1)
            set (gcf,'Position',[200,100,600,1500], 'color','w')     
            sub_cluster = clus(k: min(length(clus),k+50-1));
            for j = 1:length(sub_cluster)     

    % delay map    
            n = floor(length(sub_cluster).^0.5) +1;
            signal_length = size(obj.model{sub_cluster(j)},3);    
            subplot(n,n,j)
    %rescaling        
            min_val = 0;
            val = reshape(delay(:,:,(cluster{i}(j))),1,[]);
            val = sort(val,'descend');
            abcd = val(find(val<100));
            max_val = abcd(1);
        
            
            
            
            colormap([jet(256);gray(2)]);
            h(j) = imagesc(delay(:,:,(sub_cluster(j))));
              for abc = 1:18
                    for efg = 1:20
                        if (mask(abc,efg,(sub_cluster(j)))==0)
                            C(abc,efg,j) = 258;
                        else 
                            m = 255; 
                            C(abc,efg,j) = min(m,floor(m*(delay(abc,efg,(sub_cluster(j)))-min_val)/(max_val-min_val))+1); 

                        end
                    end
              end
            if sum(sum(mask(:,:,(sub_cluster(j))))) ~=360    
            set(h(j),'CData',C(:,:,j))
            caxis([min(min(C(:,:,j))),max(max(C(:,:,j)))]) 

            else
            set(h(j),'CData',C(:,:,j))
            caxis([min(min(C(:,:,j))),max(max(C(:,:,j)))+2])   
            end
            
        %     shading interp
            axis off  
            title(num2str(sub_cluster(j)));
            end
        
            subplot(n,n,j+1)
            
            %rescaling
         val_cluster = [];
        for j = 1:length(cluster{i})   
            val = reshape(delay(:,:,(cluster{i}(j))),1,[]);
            val_cluster = cat(2,val,val_cluster);
        end
         val_cluster = sort(val_cluster,'descend');
         min_val = 0;
         abcd = val_cluster(find(val_cluster<100));
         max_val = abcd(floor(length(abcd)*0.05)+1);
            
            
            
            label_vector = floor((0:1/5*(max_val-min_val): (max_val-min_val)));
            h2 = colorbar('XTickLabel',{num2str(label_vector(1)),num2str(label_vector(2)),...
            num2str(label_vector(3)),num2str(label_vector(4)),num2str(label_vector(5)),num2str(label_vector(6))},'location','southoutside');
    
            axis off
            set(h2,'position',[.04 .05 .92 .05])
     
     
     
    % output
          file_name = num2str(i);
          mkdir('./test24/');
   
          file_N = ['./test24/',file_name,'_',num2str((k-1)/50+1)];
          hgexport(figure (1),file_N)
          close figure 1
            
        end
    end
        
        
    
end

