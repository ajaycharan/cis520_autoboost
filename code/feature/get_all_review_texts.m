function [texts] = get_all_review_texts(metadata)
%GET_ALL_REVIEW_TEXTS Extract all review text from metadata

texts = cell(numel(metadata),1);

for i=1:numel(metadata)    
    % pull out of structure
    txt = metadata(i).text;
    texts{i} = txt;
end
end
