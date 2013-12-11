function print_msg(method, type)
msg_str   = sprintf('%s tuning model: %s', type, method);
sep_str     = repmat('=', 1, length(msg_str));
fprintf('%s\n%s\n%s\n', sep_str, msg_str, sep_str)
end