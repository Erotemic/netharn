from os.path import exists
import ubelt as ub


class CacheStamp(object):
    """
    Writes a file that marks that a procedure has been done by writing a
    "stamp" file to disk. Removing the stamp file will force recomputation.
    However, removing or changing the result of the computation may not trigger
    recomputation unless specific handling is done or the expected "product"
    of the computation is a file and registered with the stamper.  If quick is
    True, we only check if the product exists, and we ignore its hash.

    Args:
        fname (str):
            name of the stamp file

        cfgstr (str):
            configuration associated with the stamped computation.  A common
            pattern is to call `ub.hash_data` on a dependency list.

        dpath (str):
            where to store the cached stamp file

        product (str, optional):
            path that we expect the computation to produce. If specified its
            hash is stored.
            TODO: should we allow product to be a list of the computation
                produces more than one file?

        quick (bool):
            if False and product was specified, we use the product hash to
            check if it is expired.

    Example:
        >>> import ubelt as ub
        >>> from os.path import join
        >>> # Stamp the computation of expensive-to-compute.txt
        >>> dpath = ub.ensure_app_resource_dir('netharn', 'test-cache-stemp')
        >>> product = join(dpath, 'expensive-to-compute.txt')
        >>> self = CacheStamp('somedata', 'someconfig', dpath, product)
        >>> if self.expired():
        >>>     ub.writeto(product, 'very expensive')
        >>>     self.renew()
        >>> assert not self.expired()
        >>> # corrupting the output will not expire in quick mode
        >>> ub.writeto(product, 'corrupted')
        >>> assert not self.expired()
        >>> self.quick = False
        >>> # but it will if we are not in quick mode
        >>> assert self.expired()
        >>> # deleting the product will cause expiration
        >>> self.quick = True
        >>> ub.delete(product)
        >>> assert self.expired()
    """
    def __init__(self, fname, cfgstr, dpath, product=None, quick=True):
        self.cacher = ub.Cacher(fname, cfgstr=cfgstr, dpath=dpath)
        self.product = product
        self.quick = quick

    def _get_certificate(self):
        """
        Returns the stamp certificate if it exists
        """
        certificate = self.cacher.tryload()
        return certificate

    def expired(self):
        """
        Check to see if a previously existing stamp is still valid and if the
        expected result of that computation still exists.
        """
        certificate = self._get_certificate()
        if certificate is None:
            # We dont even have a certificate, so we are expired
            is_expired = True
        elif self.product is None:
            # We dont have a product to check, so assume not expired
            # TODO: we could check the timestamp in the cerficiate
            is_expired = False
        elif not exists(self.product):
            # We are expired if the expected product does not exist
            is_expired = True
        elif self.quick:
            # Assume that the product hash is the same.
            is_expired = False
        else:
            # We are expired if the hash of the existing product data
            # does not match the expected hash in the certificate
            product_hash = ub.hash_data(self.product)
            certificate_hash = certificate['product_hash']
            is_expired = not product_hash.startswith(certificate_hash)
        return is_expired

    def renew(self):
        """
        Signal that the product has been recomputed and we should recertify it.
        """
        certificate = {
            'timestamp': ub.timestamp(),
            'product': self.product,
        }
        if self.product is not None:
            if not exists(self.product):
                raise IOError(
                    'The stamped product must exist: {}'.format(self.product))
            certificate['product_hash'] = ub.hash_file(self.product)
        self.cacher.save(certificate)


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_cachestamp CacheStamp.renew
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
